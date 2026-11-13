"""Overnight top-up: keep improving the assistant's model on a normal PC (CPU is enough), safely.

Starts from the model the assistant uses now, trains a short round (default 200 steps, about 2 h on a 4-core CPU)
on your Accept/Edit feedback plus a sample of the existing training data (so it does not forget), then scores
old and new model on the same 20 everyday requests (eval/plain_requests.jsonl). The new model replaces the old
one ONLY if it scores better; the old one is always kept in nanochat_base/old_model_<date>.

Run inside WSL after setup_wsl.sh (the --cpu setup is enough):
    ~/ooa-train-venv/bin/python "/mnt/f/.../rpa model/training/local/topup_local.py"
    ... --check            only show what would be used
    ... --steps 100        shorter round
Needs in "rpa model": nanochat_base/ (the assistant's model + tokenizer), nanochat/, platform/, and sft/ with the
training data copied from Google Drive (ooa-nanochat/sft: natural_train.jsonl, core_train.jsonl, synth_train.jsonl,
...). Feedback is read from assistant/assistant_data/feedback.jsonl. Restart the assistant to use a new model.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA_DEFAULT = HERE.parent.parent
# file in the top-up mix -> (source file in sft/, rows to sample; None = all)
MIX = {
    "natural_train.jsonl": ("natural_train.jsonl", 3000),
    "core_train.jsonl": ("core_train.jsonl", None),
    "chains_train.jsonl": ("chains_train.jsonl", None),        # multi-step chains across tools: all of them, every round
    "combos_train.jsonl": ("combos_train.jsonl", None),        # many different combinations, loops, conditions: all
    "notask_train.jsonl": ("notask_train.jsonl", None),        # questions / chat -> a short text reply, not a flow
    "tools_train.jsonl": ("tools_train.jsonl", None),          # business apps (Google, Microsoft 365, Asana ...): all
    "scenarios_train.jsonl": ("scenarios_train.jsonl", None),  # whole business processes (3-8 steps, loops, ifs): all
    "synth_train.jsonl": ("synth_train.jsonl", 2000),
    "synth_pattern.jsonl": ("synth_pattern.jsonl", 800),
    "train_flows.jsonl": ("train_flows.jsonl", None),
    "train_catalog.jsonl": ("train_catalog.jsonl", 400),
}
REPEAT = {"notask_train.jsonl": 2}                            # small file: shown twice so the habit sticks


def newest(d8):
    best = None
    for f in d8.glob("model_*.pt"):
        m = re.match(r"model_(\d+)\.pt$", f.name)
        if m and (d8 / f"meta_{m.group(1)}.json").exists():
            s = int(m.group(1))
            if best is None or s > best:
                best = s
    return best


def ident(path):
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


def read_rows(p):
    with open(p, encoding="utf-8") as f:
        return [l for l in f if l.strip()]


TEST_SETS = [("plain_requests.jsonl", "set 1 (20 everyday requests)"), ("plain_requests_2.jsonl", "set 2 (15 more, never tuned on)"),
             ("plain_requests_3.jsonl", "set 3 (whole business processes, held out from the use-case data)")]


# Modules for focused rounds (--focus): a row belongs to a module by what its answer does.
GROUPS = {"files": ("ooa.file/", "ooa.folder/"), "tables": ("ooa.csv/", "ooa.datatable/"), "excel": ("ooa.excel/",),
          "word": ("ooa.word/",), "pdf": ("ooa.pdf/",), "browser": ("ooa.browser/",), "email": ("ooa.email/",),
          "apps": tuple(f"ooa.{t}/" for t in ("google", "microsoft365", "asana", "azure_devops", "github", "docusign",
                                              "box", "salesforce", "slack", "snowflake", "databricks", "jira", "servicenow",
                                              "zendesk", "okta", "workday", "exchange"))}
MODULES = ["notask", *GROUPS, "chains", "loops", "conditions"]


def flow_actions(answer):
    """Every action id in a flow answer (nested too), or None when the answer is not a flow."""
    if not answer.lstrip().startswith("{"):
        return None
    try:
        flow = json.loads(answer)
    except Exception:
        return None
    out = []
    def walk(nodes):
        for n in nodes or []:
            if isinstance(n, dict):
                if isinstance(n.get("action"), str):
                    out.append(n["action"])
                for k in ("children", "else", "finally"):
                    walk(n.get(k) if isinstance(n.get(k), list) else [])
                for c in n.get("catch") or []:
                    if isinstance(c, dict):
                        walk(c.get("children"))
    walk(flow.get("steps") if isinstance(flow, dict) else None)
    return out


def module_of(row_text):
    """The module a training row teaches, or None (catalog questions, mixed long flows ...)."""
    try:
        msgs = json.loads(row_text)["messages"]
    except Exception:
        return None
    acts = flow_actions(msgs[-1].get("content", ""))
    if acts is None:
        return "notask"
    if "core/loop" in acts:
        return "loops"
    if "core/if" in acts:
        return "conditions"
    plain = [a for a in acts if not a.startswith("core/")]
    groups = {g for a in plain for g, pre in GROUPS.items() if a.startswith(pre)}
    if len(groups) == 1:
        return groups.pop()
    if len(groups) >= 2:
        return "chains"
    return None


def test_rows(rpa):
    out = []
    for i, (name, _) in enumerate(TEST_SETS, 1):
        p = rpa / "eval" / name
        if p.exists():
            out += [dict(json.loads(l), set=i) for l in read_rows(p)]
    return out


def summary(details):
    """'set 1: right 8/20, verified 17/20 | set 2: ...' and the totals."""
    parts = []
    for i, (_, label) in enumerate(TEST_SETS, 1):
        d = [x for x in details if x.get("set", 1) == i]
        if d:
            parts.append(f"set {i}: right {sum(x['right'] for x in d)}/{len(d)}, verified {sum(x['verified'] for x in d)}/{len(d)}")
    return " | ".join(parts)


def score(rpa, model_dir, catalog, port, log, rows=None):
    """Start the assistant on model_dir and score every test request (both sets), or the given rows:
    (right, verified, details). A row with notask=True is right when the model answers with text, not a flow."""
    sys.path.insert(0, str(rpa / "assistant"))
    import basic_test as B
    data = tempfile.mkdtemp(prefix="ooa_eval_")
    cmd = [sys.executable, str(rpa / "assistant" / "app.py"), "--model-dir", str(model_dir), "--nanochat",
           str(rpa / "nanochat"), "--platform", str(rpa / "platform"), "--catalog", str(catalog),
           "--data-dir", data, "--port", str(port)]
    with open(log, "a", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(rpa / "assistant"))
    try:
        for _ in range(300):                                    # model load can take a minute or two on a CPU
            time.sleep(2)
            if p.poll() is not None:
                raise SystemExit(f"the assistant did not start for scoring (see {log})")
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/status",
                                                              headers={"X-OOA-Assistant": "1"}), timeout=5)
                break
            except Exception:
                continue
        else:
            raise SystemExit(f"the assistant did not answer within 10 minutes (see {log})")
        rows = test_rows(rpa) if rows is None else rows
        right = verified = 0
        details = []
        for r in rows:
            d = B.build(port, r["prompt"])
            got = B.actions((d.get("flow") or {}).get("steps"))
            ok = bool(d.get("verified"))
            if r.get("notask"):
                rt = not ok and any("not read this as a task" in str(x) for x in d.get("problems") or [])
            else:
                rt = ok and B.in_order(r["must"], got)
            verified += ok
            right += rt
            details.append({"request": r["prompt"], "verified": ok, "right": rt, "set": r["set"],
                            "actions": got, "must": r["must"], "problems": (d.get("problems") or [])[:3]})
        return right, verified, details
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()
        shutil.rmtree(data, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(RPA_DEFAULT), help='the "rpa model" folder of the assistant')
    ap.add_argument("--mode", choices=["topup", "full"], default="topup",
                    help="topup: short round from the current model (CPU ok); full: the Colab recipe from the "
                         "pretrained base model (needs nanochat_base/base_checkpoints; GPU, or about a day on a CPU)")
    ap.add_argument("--steps", type=int, default=None, help="training steps (32,768 tokens each); default 200 topup, 2500 full")
    ap.add_argument("--lr", type=float, default=None, help="learning rate fraction; default 0.3 topup (it only adjusts), 0.8 full")
    ap.add_argument("--device-batch", type=int, default=2, choices=[1, 2, 4, 8, 16], help="lower = less RAM, same result (default 2)")
    ap.add_argument("--feedback-repeat", type=int, default=5, help="how often each piece of your feedback is shown")
    ap.add_argument("--port", type=int, default=8799, help="port for the scoring runs (not the assistant's 8765)")
    ap.add_argument("--max-forget", type=int, default=1,
                    help="install only if at most this many test requests the current model gets right become wrong")
    ap.add_argument("--focus", choices=MODULES, default=None,
                    help="focused round: mostly this module's examples plus a replay of everything else; some of its"
                         " examples are held out and scored too")
    ap.add_argument("--focus-share", type=float, default=0.6, help="share of the round's rows from the focus module")
    ap.add_argument("--holdout", type=int, default=12, help="focus-module examples held out for scoring")
    ap.add_argument("--auto-steps", action="store_true",
                    help="focused round: size the round to about --passes passes over its rows (at most --steps)")
    ap.add_argument("--passes", type=float, default=2.0)
    ap.add_argument("--check", action="store_true", help="only show what would be used")
    a = ap.parse_args()
    full = a.mode == "full"
    a.steps = a.steps or (2500 if full else 200)
    a.lr = a.lr if a.lr is not None else (0.8 if full else 0.3)
    os.environ.setdefault("PYTHONUTF8", "1")          # child processes (training, scoring) write UTF-8 logs on Windows too
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")   # a Windows/company proxy must not catch the local scoring calls
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        pass

    rpa = Path(a.rpa).expanduser().resolve()
    base = rpa / "nanochat_base"
    d8 = base / "chatsft_checkpoints" / "d8"
    sft = rpa / "sft"
    clean = rpa / "sft_clean"                  # written by training/data_filter.py (MODE "big" runs it)
    rep = clean / "filter_report.json"
    if rep.exists() and all(f.stat().st_mtime <= rep.stat().st_mtime for f in sft.glob("*.jsonl")):
        print("training data   : the cleaned copy in", clean, "(data_filter.py)")
        sft = clean
    elif rep.exists():
        print("training data   : sft/ changed after the last cleaning; using sft/ as it is (run data_filter.py again)")
    work = rpa / f"topup_work_{time.strftime('%Y%m%d_%H%M%S')}"   # a fresh folder each round (Windows can leave files locked)
    fb = rpa / "assistant" / "assistant_data" / "feedback.jsonl"
    catalog = sft / "catalog.jsonl" if (sft / "catalog.jsonl").exists() else rpa / "data" / "catalog" / "catalog.jsonl"
    schema = rpa / "platform" / "contracts" / "flow-schema" / "flow.v1.json"
    history = base / "topup_history.jsonl"

    step = newest(d8)
    problems = []
    if step is None:
        problems.append(f"no model_<N>.pt with its meta in {d8}")
    bd8 = base / "base_checkpoints" / "d8"
    bstep = newest(bd8) if full else None
    if full and bstep is None:
        problems.append(f"full mode needs the pretrained base model in {bd8} (model_<N>.pt + meta_<N>.json + optim)")
    for f in (base / "tokenizer" / "tokenizer.pkl", base / "tokenizer" / "token_bytes.pt", schema,
              rpa / "nanochat" / "scripts" / "chat_sft.py", rpa / "assistant" / "app.py", rpa / "eval" / "plain_requests.jsonl"):
        if not f.exists():
            problems.append(f"missing {f}")
    have = {k: (sft / src).exists() for k, (src, _) in MIX.items()}
    fb_rows = len(read_rows(fb)) if fb.exists() else 0
    print("assistant folder:", rpa)
    print("current model   :", f"{d8}/model_{step:06d}.pt" if step is not None else "-")
    print("training data   :", ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in have.items()))
    print("your feedback   :", f"{fb_rows} answers rated" if fb_rows else "none yet (Accept / Edit / Reject in the Build tab)")
    if not any(have.values()) and not fb_rows:
        problems.append(f"nothing to learn from: copy the Drive folder ooa-nanochat/sft into {sft}, or rate some answers")
    for p_ in problems:
        print("  !", p_)
    if problems:
        raise SystemExit("\nNot ready: fix the lines marked ! above.")
    if a.check:
        print("\nREADY. Run again without --check (best overnight: about 2 h for 200 steps on a 4-core CPU).")
        return

    t0 = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = rpa / f"topup_{stamp}.log"
    model, meta = d8 / f"model_{step:06d}.pt", d8 / f"meta_{step:06d}.json"
    print(f"\n== 1/4 preparing (log: {log})")
    for old in rpa.glob("topup_work_*"):                       # leftovers of earlier rounds
        shutil.rmtree(old, ignore_errors=True)
    if os.name == "nt":                                         # keep the PC awake while this process runs
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass
    wb = work / "nanochat_base"
    (wb / "base_checkpoints" / "d8").mkdir(parents=True)
    shutil.copytree(base / "tokenizer", wb / "tokenizer")
    if full:                                                    # train from the pretrained base model, like Colab
        for f in bd8.iterdir():
            if re.match(rf"(model|meta|optim)_{bstep:06d}", f.name):
                shutil.copy2(f, wb / "base_checkpoints" / "d8" / f.name)
    else:
        shutil.copy2(model, wb / "base_checkpoints" / "d8" / model.name)      # train FROM the current model
        shutil.copy2(meta, wb / "base_checkpoints" / "d8" / meta.name)
    ws = work / "sft"
    ws.mkdir()
    rng = random.Random(stamp)
    val = []
    if full:                                                    # the whole data folder, as Colab cell 7 uses it
        for f in sft.glob("*.jsonl"):
            if f.name not in ("catalog.jsonl", "feedback_train.jsonl"):
                shutil.copy2(f, ws / f.name)
        val = read_rows(ws / "val.jsonl") if (ws / "val.jsonl").exists() else []
        for need in ("train_flows.jsonl", "train_catalog.jsonl", "synth_train.jsonl", "synth_pattern.jsonl"):
            if not (ws / need).exists():
                (ws / need).write_text("", encoding="utf-8")
    picked, module_rows = {}, []
    for name, (src, n) in ({} if full else MIX).items():
        rows = read_rows(sft / src) if (sft / src).exists() else []
        rng.shuffle(rows)
        if name == "natural_train.jsonl" and len(rows) > 200:
            val, rows = rows[:100], rows[100:]                 # held out: watched during training, never trained on
        picked[name] = rows if (n is None or a.focus) else rows[:n]
    if a.focus and picked:
        focus = {k: [r for r in v if module_of(r) == a.focus][:MIX[k][1] or None] for k, v in picked.items()}
        other = {k: [r for r in v if module_of(r) != a.focus] for k, v in picked.items()}
        pool = [(k, r) for k, v in focus.items() for r in v]
        rng.shuffle(pool)
        held = pool[:a.holdout]                                # never trained on: scored to see if this module improved
        held_set = {id(r) for _, r in held}
        for _, r in held:
            msgs = json.loads(r)["messages"]
            acts = flow_actions(msgs[-1]["content"])
            module_rows.append({"prompt": msgs[0]["content"] if msgs[0]["role"] == "user" else msgs[1]["content"],
                                "must": [x for x in (acts or []) if x.split("/")[-1] not in ("open", "save", "close")],
                                "notask": acts is None, "set": "module"})
        n_focus = sum(len(v) for v in focus.values()) - len(held)
        n_other = int(n_focus * (1 - a.focus_share) / max(a.focus_share, 0.05))
        total_other = sum(min(len(v), MIX[k][1] or len(v)) for k, v in other.items()) or 1
        for k in picked:
            keep_other = other[k][:max(0, round(n_other * min(len(other[k]), MIX[k][1] or len(other[k])) / total_other))]
            picked[k] = [r for r in focus[k] if id(r) not in held_set] + keep_other
        print(f"  focus {a.focus}: {n_focus} rows of it + {sum(len(v) for v in picked.values()) - n_focus} replay rows,"
              f" {len(module_rows)} held out for scoring")
        if n_focus < 20:
            raise SystemExit(f"too few {a.focus} examples in the data ({n_focus}); nothing changed")
    for name, rows in picked.items():
        (ws / name).write_text("".join(rows * REPEAT.get(name, 1)), encoding="utf-8")
    if a.focus and a.auto_steps:     # about 2 passes: enough to learn the module, not enough to memorise it
        tokens = sum(len(r) for v in picked.values() for r in v) / 3.2        # ~3.2 characters per token in flow JSON
        a.steps = max(10, min(a.steps, round(tokens * a.passes / 32768)))
        print(f"  round size: {a.steps} steps (about {a.passes:g} passes over {sum(len(v) for v in picked.values())} rows)")
    if not val:
        val = read_rows(sft / "val.jsonl")[:150] if (sft / "val.jsonl").exists() else []
    if fb_rows:
        r = subprocess.run([sys.executable, str(rpa / "training" / "ingest_feedback.py"), str(fb), str(work / "fb"),
                            str(catalog), str(schema)], capture_output=True, text=True)
        print("  feedback:", (r.stdout.strip() or r.stderr.strip()[-300:]))
        fb_train = work / "fb" / "feedback_train.jsonl"
        if fb_train.exists():
            (ws / "feedback_train.jsonl").write_text("".join(read_rows(fb_train) * a.feedback_repeat), encoding="utf-8")
            if not val:
                val = read_rows(fb_train)[:20]
    if not val:
        raise SystemExit("no validation rows: copy the Drive sft folder (it has natural_train.jsonl) into " + str(sft))
    (ws / "val.jsonl").write_text("".join(val), encoding="utf-8")
    shutil.copy2(catalog, ws / "catalog.jsonl")

    if full:
        print(f"== 2/4 full training: {a.steps} steps from the pretrained base model (GPU: about 1 h; CPU: about a day)")
        cmd = [sys.executable, str(HERE / "train_local.py"), "--work", str(work), "--nanochat", str(rpa / "nanochat"),
               "--steps", str(a.steps), "--device-batch", str(a.device_batch), "--init-lr-frac", str(a.lr), "--allow-cpu"]
    else:
        print(f"== 2/4 training {a.steps} steps from model step {step} (CPU: about {a.steps * 38 // 60} min on 4 cores)")
        cmd = [sys.executable, str(HERE / "train_local.py"), "--work", str(work), "--nanochat", str(rpa / "nanochat"),
               "--steps", str(a.steps), "--device-batch", str(a.device_batch), "--epochs", "1", "--flow-repeat", "1",
               "--natural-repeat", "1", "--core-repeat", "1", "--smoltalk", "0", "--init-lr-frac", str(a.lr),
               "--eval-every", str(max(50, a.steps // 2)), "--allow-cpu"]
    with open(log, "a", encoding="utf-8") as lf:      # full output to the log, a progress line every 10 steps here
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            m = re.match(r"step (\d+) \(([\d.]+)%\) \| loss: ([\d.]+).*?dt: ([\d.]+)ms", line)
            if m and (int(m.group(1)) % 10 == 0 or int(m.group(1)) == 1):
                left = (a.steps - int(m.group(1))) * float(m.group(4)) / 60000
                print(f"   step {int(m.group(1))}/{a.steps} | loss {m.group(3)} | {float(m.group(4)) / 1000:.0f} s per step"
                      f" | about {left:.0f} min to go ({time.strftime('%H:%M')})")
        r = p
        r.returncode = p.wait()
    if r.returncode != 0:
        raise SystemExit(f"training failed (see the end of {log}); the assistant's model was not touched")
    cd8 = work / "nanochat_base" / "chatsft_checkpoints" / "d8"
    cstep = newest(cd8)
    if cstep is None:
        raise SystemExit(f"training produced no model (see {log}); nothing changed")

    n_tests = len(test_rows(rpa))
    print(f"== 3/4 scoring old and new model on the {n_tests} test requests")
    hist = [json.loads(l) for l in read_rows(history)] if history.exists() else []
    cur_id = ident(model)
    cached = next((h for h in reversed(hist) if h.get("kept_id") == cur_id and h.get("tests") == n_tests
                   and h.get("kept_details")), None)
    if cached:
        b_right, b_ver = cached["kept_score"]
        b_details = cached["kept_details"]
        print(f"  current model: {summary(b_details)}  (from the last round)")
    else:
        b_right, b_ver, b_details = score(rpa, base, catalog, a.port, log)
        print(f"  current model: {summary(b_details)}")
    n_right, n_ver, details = score(rpa, work / "nanochat_base", catalog, a.port, log)
    print(f"  new model    : {summary(details)}")
    b_mod = n_mod = 0
    if module_rows:                                            # the focus module's held-out examples, both models
        b_mod = score(rpa, base, catalog, a.port, log, module_rows)[0]
        n_mod, _, mod_details = score(rpa, work / "nanochat_base", catalog, a.port, log, module_rows)
        print(f"  {a.focus} (held-out examples): current model {b_mod}/{len(module_rows)} right, new model {n_mod}/{len(module_rows)}")

    was_right = {d["request"] for d in b_details if d.get("right")}
    forgot = [d["request"] for d in details if d["request"] in was_right and not d.get("right")]
    if forgot:
        print(f"  forgot {len(forgot)} request(s) the current model got right:", "; ".join(f[:50] for f in forgot[:5]))
    if module_rows:      # focused round: better on its module AND not worse on the test requests
        gain = (n_right + n_mod, n_ver) > (b_right + b_mod, b_ver) and n_right >= b_right
    else:
        gain = (n_right, n_ver) > (b_right, b_ver)
    better = gain and len(forgot) <= a.max_forget
    reason = ("new model is BETTER: installing it" if better else
              f"new model forgot {len(forgot)} request(s) it used to get right (limit {a.max_forget}): keeping the current one"
              if gain else "new model is not better: keeping the current one")
    print("== 4/4", reason)
    rec = {"time": stamp, "trained_steps": cstep, "from_step": step, "feedback_rows": fb_rows,
           "old_score": [b_right, b_ver], "new_score": [n_right, n_ver], "installed": better, "details": details,
           "forgot": forgot, "lr": a.lr, "focus": a.focus, "module_score": [b_mod, n_mod, len(module_rows)],
           "tests": n_tests}
    if better:
        new_step = cstep if full else step + cstep              # the old model is moved out of d8 below in any case
        keep = base / f"old_model_{stamp}"
        keep.mkdir()
        for f in d8.iterdir():
            if re.match(r"(model|meta|optim)_\d+", f.name):
                shutil.move(str(f), keep / f.name)
        shutil.copy2(cd8 / f"model_{cstep:06d}.pt", d8 / f"model_{new_step:06d}.pt")
        m = json.loads((cd8 / f"meta_{cstep:06d}.json").read_text(encoding="utf-8"))
        m["step"] = new_step
        m["ooa_topup"] = {"time": stamp, "mode": a.mode, "from_step": bstep if full else step, "steps": cstep,
                          "lr": a.lr, "score": [n_right, n_ver]}
        (d8 / f"meta_{new_step:06d}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        rec.update(kept=f"model_{new_step:06d}.pt", kept_score=[n_right, n_ver], old_kept_in=str(keep), kept_details=details)
        rec["kept_id"] = ident(d8 / f"model_{new_step:06d}.pt")
        print(f"  installed model_{new_step:06d}.pt; the previous model is kept in {keep}")
        print("  restart the assistant (close its window, start it again) to use the new model")
    else:
        rec.update(kept=model.name, kept_score=[b_right, b_ver], kept_id=cur_id, kept_details=b_details)
    with open(history, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min. History: {history}")


if __name__ == "__main__":
    main()
