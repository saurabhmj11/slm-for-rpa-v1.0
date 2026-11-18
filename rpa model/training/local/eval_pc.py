"""Score a newly trained model against the one the assistant uses now, the way Colab cell 8 scores the full
assistant, and install the new one only if it is better.

    python eval_pc.py --rpa DIR --new FOLDER --data DIR [--per-set 6] [--minutes 60] [--install]

--new: the folder with the new model_<N>.pt + meta_<N>.json (train_pc.py's last chunk, or Drive's newest model).
--data: the rendered data it was trained on (sft_render): its catalog.jsonl and examples.jsonl are the new model's
context (the same files the server uses). The installed model is scored with what the assistant gives it now
(rpa/sft/catalog.jsonl, and examples.jsonl beside it when there is one, else plain prompts).

Test sets (rpa/eval): set 1-3 (plain_requests*.jsonl: right = verified flow with the must actions in order) and the
beginner set (beginner_requests.jsonl, eval/beginner_score.py: a flow when one can be built, a question about the
missing detail when only the user knows it). Each request goes through the full assistant (serving/ooa_server.Router
with the catalog lock, repair, verify and up to --retries self-corrections, as in cell 8). On a CPU a request takes
from seconds to a few minutes, so a sample is scored: --per-set requests of each set (the same ones for both models,
spread evenly over the set, taken in turn from each set) and at most --minutes per model; the comparison uses the
requests both models finished. Scores of the installed model are cached (local_run/eval_cache.json).

Better = more requests right in total, ties broken by more verified flows. Install: the old model and its catalog /
examples move to nanochat_base/old_model_<date>; the new model goes to nanochat_base/chatsft_checkpoints/d8 and its
catalog.jsonl + examples.jsonl to rpa/sft/ (where assistant/app.py and start_assistant.bat read them, so the app
uses the context + plan format). History: nanochat_base/topup_history.jsonl.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = [("plain_requests.jsonl", "set 1"), ("plain_requests_2.jsonl", "set 2"),
        ("plain_requests_3.jsonl", "set 3"), ("beginner_requests.jsonl", "beginner")]


def newest(d):
    best = None
    for f in Path(d).glob("model_*.pt") if Path(d).exists() else []:
        m = re.match(r"model_(\d+)\.pt$", f.name)
        if m and (Path(d) / f"meta_{m.group(1)}.json").exists():
            best = max(best or 0, int(m.group(1)))
    return best


def sample_rows(eval_dir, per_set):
    """[(id, set label, row)] taken in turn from each set, per_set from each (0 = all), spread evenly."""
    per = []
    for name, label in SETS:
        p = Path(eval_dir) / name
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(rows) if per_set <= 0 else min(per_set, len(rows))
        idx = sorted({int(i * len(rows) / n) for i in range(n)})
        per.append([(f"{label}:{i}", label, rows[i]) for i in idx])
    out = []
    for k in range(max((len(x) for x in per), default=0)):
        out += [x[k] for x in per if k < len(x)]
    return out


def file_id(p):
    p = Path(p)
    if not p.exists():
        return "-"
    st = p.stat()
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"


# ---------------------------------------------------------------------------------- scoring (one model per process)
def score_main(a):
    """Load one model, run the rows through the full assistant, write {id: result} to --out as it goes."""
    rpa = Path(a.rpa)
    for p in (rpa / "training", rpa / "serving", rpa / "assistant", rpa / "eval"):
        sys.path.insert(0, str(p))
    import torch
    import ooa_server as S
    import ooa_reward as RW
    import beginner_score as B
    from model_backend import NanochatLocal
    torch.set_num_threads(a.threads or os.cpu_count() or 4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    local = NanochatLocal(a.model_root, str(rpa / "nanochat"), source="sft", device=dev, catalog=a.catalog)
    router = S.Router(a.catalog, a.platform, S.Policy(allow_escalation=False, local_retries=a.retries), local, None,
                      a.examples or None, a.format)
    rows = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    out = json.loads(Path(a.out).read_text(encoding="utf-8")) if Path(a.out).exists() else {}
    end = time.time() + a.minutes * 60
    print(f"  {a.name}: step {local.info.get('step')} | prompts {router.format()} | trained window "
          f"{local.info.get('trained_window')} | {len(rows)} requests, at most {a.minutes:g} min", flush=True)

    def actions(t):
        return RW.actions_of(json.loads(t))
    for rid, label, row in rows:
        if rid in out:
            continue
        if time.time() > end:
            print(f"  {a.name}: time limit reached", flush=True)
            break
        t0 = time.time()
        try:
            res = router.run(row["prompt"])
        except Exception as ex:                                  # a crash counts as a wrong answer, never stops
            res = {"verified": False, "error": str(ex)[:200]}
        if "kind" in row:
            sc = B.score_router(row, res, actions)
            right, outcome = bool(sc["right"]), sc["outcome"]
        else:
            got = actions(res["flow"]) if res.get("verified") else []
            right, outcome = bool(res.get("verified")) and B.in_order(row["must"], got), ""
        got = actions(res["flow"]) if res.get("verified") else []
        out[rid] = {"set": label, "right": right, "verified": bool(res.get("verified")), "outcome": outcome,
                    "seconds": round(time.time() - t0, 1), "actions": got[:8],
                    "question": (res.get("question") or "")[:120]}
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"  {a.name}: {'OK   ' if right else ('VALID' if res.get('verified') else 'FAIL ')} {out[rid]['seconds']:>6}s "
              f"{label:8s} {row['prompt'][:60]}", flush=True)
    return 0


def run_scoring(rpa, name, model_root, catalog, examples, fmt, rows, minutes, retries, platform, out, threads=0):
    rows_f = Path(out).with_suffix(".rows.json")
    rows_f.write_text(json.dumps(rows), encoding="utf-8")
    cmd = [sys.executable, "-u", str(HERE / "eval_pc.py"), "score", "--rpa", str(rpa), "--name", name,
           "--model-root", str(model_root), "--catalog", str(catalog), "--examples", str(examples or ""), "--format", fmt,
           "--rows", str(rows_f), "--minutes", str(minutes), "--retries", str(retries), "--platform", str(platform),
           "--out", str(out), "--threads", str(threads)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    res = json.loads(Path(out).read_text(encoding="utf-8")) if Path(out).exists() else {}
    if r.returncode != 0 and not res:
        raise SystemExit(f"scoring {name} failed (see above)")
    return res


def summarize(res, ids):
    by = {}
    for i in ids:
        r = res[i]
        s = by.setdefault(r["set"], [0, 0, 0])
        s[0] += r["right"]
        s[1] += r["verified"]
        s[2] += 1
    right = sum(v[0] for v in by.values())
    ver = sum(v[1] for v in by.values())
    text = " | ".join(f"{k}: {v[0]}/{v[2]} right ({v[1]} verified)" for k, v in by.items())
    return right, ver, text


# ---------------------------------------------------------------------------------- install
def current_pack(rpa):
    """What the assistant uses now: (model root, catalog, examples or None, prompt format)."""
    rpa = Path(rpa)
    cat = rpa / "sft" / "catalog.jsonl"
    cat = cat if cat.exists() else rpa / "data" / "catalog" / "catalog.jsonl"
    ex = cat.parent / "examples.jsonl"
    return rpa / "nanochat_base", cat, (ex if ex.exists() else None), ("context" if ex.exists() else "plain")


def install(rpa, new_dir, new_step, data, info, stamp, log=print):
    rpa, data = Path(rpa), Path(data)
    base = rpa / "nanochat_base"
    d8 = base / "chatsft_checkpoints" / "d8"
    keep = base / f"old_model_{stamp}"
    old = newest(d8)
    final_step = info.get("install_step") or new_step
    if d8.exists() and any(d8.iterdir()):
        keep.mkdir(parents=True, exist_ok=True)
        for f in d8.iterdir():
            if re.match(r"(model|meta|optim)_\d+", f.name):
                shutil.move(str(f), keep / f.name)
        for f in ("catalog.jsonl", "examples.jsonl"):
            if (rpa / "sft" / f).exists():
                shutil.copy2(rpa / "sft" / f, keep / f)
        log(f"  the previous model (step {old}) and its catalog / examples are kept in {keep}")
    d8.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(new_dir) / f"model_{new_step:06d}.pt", d8 / f"model_{final_step:06d}.pt")
    m = json.loads((Path(new_dir) / f"meta_{new_step:06d}.json").read_text(encoding="utf-8"))
    m["step"] = final_step
    m.setdefault("ooa_local", {}).update(info, installed=stamp)
    (d8 / f"meta_{final_step:06d}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    (rpa / "sft").mkdir(exist_ok=True)
    for f in ("catalog.jsonl", "examples.jsonl"):           # the model pack: the context the model was trained with
        shutil.copy2(data / f, rpa / "sft" / f)
    tok = info.get("tokenizer")
    if tok and Path(tok).exists():
        for f in ("tokenizer.pkl", "token_bytes.pt"):
            dst = base / "tokenizer" / f
            if not dst.exists() or dst.read_bytes() != (Path(tok) / f).read_bytes():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.copy2(dst, keep / f)
                shutil.copy2(Path(tok) / f, dst)
    log(f"  INSTALLED model_{final_step:06d}.pt + {rpa / 'sft'}/catalog.jsonl, examples.jsonl. Restart the assistant "
        "to use it.")
    return d8 / f"model_{final_step:06d}.pt"


def compare_main(a):
    rpa = Path(a.rpa).resolve()
    platform = Path(a.platform) if a.platform else rpa / "platform"
    st = json.loads(Path(a.run_state).read_text(encoding="utf-8")) if a.run_state and Path(a.run_state).exists() else {}
    a.new = a.new or st.get("result")
    a.data = a.data or st.get("data")
    if not a.new or not a.data:
        raise SystemExit("give --new and --data, or --run-state of a finished training run")
    if st.get("finished"):
        print(f"note: this training run was already tested on {st['finished']}; testing it again")
    data = Path(a.data).resolve()
    new_dir = Path(a.new).resolve()
    nstep = newest(new_dir)
    if nstep is None:
        raise SystemExit(f"no model_<N>.pt + meta_<N>.json in {new_dir}")
    work = rpa / "local_run" / "eval"
    shutil.rmtree(work, ignore_errors=True)
    (work / "new" / "chatsft_checkpoints" / "d8").mkdir(parents=True)
    for f in (f"model_{nstep:06d}.pt", f"meta_{nstep:06d}.json"):     # a model folder as NanochatLocal wants it
        shutil.copy2(new_dir / f, work / "new" / "chatsft_checkpoints" / "d8" / f)
    if st.get("tokenizer") and not a.tokenizer:
        a.tokenizer = st["tokenizer"]
    tok = Path(a.tokenizer) if a.tokenizer else next(
        (p for p in (new_dir.parent.parent / "tokenizer", rpa / "train_data" / "drive_model" / "tokenizer",
                     rpa / "nanochat_base" / "tokenizer") if (p / "tokenizer.pkl").exists()), None)
    if tok is None:
        raise SystemExit("no tokenizer found for the new model")
    shutil.copytree(tok, work / "new" / "tokenizer")
    rows = sample_rows(rpa / "eval", a.per_set)
    ids = [r[0] for r in rows]
    print(f"== scoring on {len(rows)} test requests ({a.per_set or 'all'} per set: sets 1-3 and the beginner set), "
          f"full assistant, at most {a.minutes:g} min per model")

    croot, ccat, cex, cfmt = current_pack(rpa)
    cstep = newest(croot / "chatsft_checkpoints" / "d8")
    old = {}
    if cstep is not None:
        cache_f = rpa / "local_run" / "eval_cache.json"
        key = hashlib.sha1(json.dumps([file_id(croot / "chatsft_checkpoints" / "d8" / f"model_{cstep:06d}.pt"),
                                       file_id(ccat), file_id(cex) if cex else "-", cfmt, a.retries]).encode()).hexdigest()[:16]
        cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
        old_f = work / "old.json"
        old_f.write_text(json.dumps({k: v for k, v in cache.get(key, {}).items() if k in ids}), encoding="utf-8")
        old = run_scoring(rpa, "installed model", croot, ccat, cex, cfmt, rows, a.minutes, a.retries, platform, old_f,
                          a.threads)
        cache[key] = dict(cache.get(key, {}), **old)
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(cache), encoding="utf-8")
    new = run_scoring(rpa, "new model", work / "new", data / "catalog.jsonl", data / "examples.jsonl", "context",
                      rows, a.minutes, a.retries, platform, work / "new.json", a.threads)

    both = [i for i in ids if i in new and (cstep is None or i in old)]
    n_right, n_ver, n_text = summarize(new, both)
    print(f"\n== results on the {len(both)} requests both models finished")
    if cstep is not None:
        o_right, o_ver, o_text = summarize(old, both)
        print(f"  installed model (step {cstep}, {cfmt} prompts): {o_right}/{len(both)} right, {o_ver} verified | {o_text}")
    else:
        o_right = o_ver = -1
        print("  installed model: none")
    print(f"  new model (step {nstep}): {n_right}/{len(both)} right, {n_ver} verified | {n_text}")
    better = bool(both) and (n_right, n_ver) > (o_right, o_ver)
    if not both:
        print("  no request finished for both models: not installing (raise --minutes)")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rec = {"time": stamp, "new": str(new_dir), "new_step": nstep, "old_step": cstep, "requests": len(both),
           "old_score": [o_right, o_ver], "new_score": [n_right, n_ver], "better": better, "installed": False}
    if better and a.install:
        info = json.loads(a.info) if a.info else {}
        if st:
            info.update(mode=st.get("mode"), from_step=st.get("start_step"), steps=st.get("steps"),
                        install_step=st.get("install_step"), drive_origin=st.get("drive_origin"))
        info["tokenizer"] = str(tok)
        rec["installed"] = str(install(rpa, new_dir, nstep, data, info, stamp))
    elif better:
        print("  the new model is BETTER (not installed: --install not given)")
    else:
        print("  the new model is NOT better: the installed one stays. The new one is kept in", new_dir)
    hist = rpa / "nanochat_base" / "topup_history.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if a.run_state and Path(a.run_state).exists():               # the training run is done: the next one starts fresh
        st = json.loads(Path(a.run_state).read_text(encoding="utf-8"))
        st["finished"], st["installed"] = stamp, rec["installed"]
        Path(a.run_state).write_text(json.dumps(st, indent=1), encoding="utf-8")
    shutil.rmtree(work / "new", ignore_errors=True)
    print(json.dumps({k: rec[k] for k in ("old_score", "new_score", "requests", "better", "installed")}))
    return 0


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        pass
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(HERE.parent.parent))
    ap.add_argument("--platform", default="")
    ap.add_argument("--retries", type=int, default=2, help="self-corrections per request (cell 8: 2)")
    ap.add_argument("--minutes", type=float, default=60, help="at most this long per model")
    ap.add_argument("--threads", type=int, default=0)
    if argv and argv[0] == "score":
        ap.add_argument("cmd")
        for k in ("--name", "--model-root", "--catalog", "--examples", "--format", "--rows", "--out"):
            ap.add_argument(k, default="")
        return score_main(ap.parse_args(argv))
    ap.add_argument("--new", default="", help="folder with the new model_<N>.pt + meta_<N>.json (default: the result "
                                              "in --run-state)")
    ap.add_argument("--data", default="", help="the rendered data the new model trained on (sft_render; default: "
                                               "the one in --run-state)")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--per-set", type=int, default=10, help="requests per test set (0 = all 139)")
    ap.add_argument("--install", action="store_true", help="install the new model when it is better")
    ap.add_argument("--info", default="", help="JSON added to the installed meta (ooa_local)")
    ap.add_argument("--run-state", default="", help="train_pc.py's state.json, marked finished afterwards")
    return compare_main(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
