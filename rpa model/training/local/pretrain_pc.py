"""Pretrain the clean base model (v6) on this PC's NVIDIA GPU: notebook MODE "pretrain". Resumable after any stop.

    python pretrain_pc.py --rpa DIR --check          the plan: profile, time and disk estimates, what is in place
    python pretrain_pc.py --rpa DIR                  build the data, train the tokenizer, pretrain, write the pack
    python pretrain_pc.py --rpa DIR --measure 20     20 steps on this GPU: tokens per second, memory, the real hours
    python pretrain_pc.py --rpa DIR --restart        throw the run in progress away (the data and tokenizer stay)

Stages (each skipped when done; state in RPA/local_run/pretrain/):
  1. data       training/pretrain/pretrain_data.py streams FineWeb-Edu (ODC-By 1.0) from Hugging Face and adds the
                owner's domain text (platform docs / code from RPA/platform, the rendered training rows of
                RPA/train_data/sft_render, the catalog), written as nanochat shards in local_run/pretrain/base/
                base_data_climbmix (nanochat's folder name; no ClimbMix in it). Or the shards Drive holds
                (nanochat_base_v6/base_data_climbmix, downloaded by data_pc.py). Deterministic: a rerun keeps them.
  2. tokenizer  nanochat's scripts/tok_train.py on those shards (16,384 tokens), or Drive's v6 tokenizer when data_pc.py
                downloaded one (--tokenizer), so the PC and Colab use the same one; tokenizer_card.json says what it
                is (sha1) and measures it on the rendered rows.
  3. pretrain   nanochat's scripts/base_train.py from the pinned commit, patched by training/pretrain/base_setup.py
                (NaN guard: a loss or validation that is not a number stops the run and nothing broken is saved;
                only the 2 newest checkpoints kept; the tokenizer's sha1 in every checkpoint; no torch.compile on
                Windows). Saved every save_every steps (pc profile: 200 steps, about 22 min): after a stop, crash or
                restart, run it again and it resumes from the newest good checkpoint (model, optimizer, data
                position), on the same learning-rate schedule.
  4. pack       the finished model as RPA/train_data/base_v6 (tokenizer + card, base_checkpoints/d8, VERSION,
                DATA_CARD.md). The current base (train_data/base) and the installed model (nanochat_base) are never
                touched. Then MODE "full" with BASE = "v6" trains the OOA SFT on it.
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
RPA_DEFAULT = HERE.parent.parent
sys.path[:0] = [str(HERE.parent / "pretrain"), str(HERE.parent), str(HERE)]
import base_setup as B                                         # noqa: E402
import sft_setup as T                                          # noqa: E402
from train_pc import (NANOCHAT_COMMIT, device_info, windows_kill_with_parent, model_health,  # noqa: E402
                      nan_value)

PACK_NAME = "base_v6"


def paths(rpa):
    run = Path(rpa) / "local_run" / "pretrain"
    work = run / "base"
    return {"run": run, "work": work, "data": work / "base_data_climbmix", "tok": work / "tokenizer",
            "ckpt": work / "base_checkpoints" / B.MODEL_TAG, "state": run / "state.json", "log": run / "pretrain.log",
            "pack": Path(rpa) / "train_data" / PACK_NAME}


def pinned_script(nano, rel):
    """A nanochat script of the pinned commit: from git when nanochat is a clone, else the file as it is (the zip /
    setup_pc ship the pinned version)."""
    nano = Path(nano)
    if (nano / ".git").exists():
        try:
            r = subprocess.run(["git", "-C", str(nano), "show", f"{NANOCHAT_COMMIT}:{rel}"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0 and r.stdout:
                return r.stdout.replace("\r\n", "\n")
        except OSError:
            pass
    return (nano / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def write_script(nano, compile_ok):
    s = B.patch_base_train(pinned_script(nano, "scripts/base_train.py"), compile_ok=compile_ok)
    out = Path(nano) / "scripts" / "base_train_ooa.py"          # base_train.py itself stays untouched
    out.write_text(s, encoding="utf-8")
    return out


def resolve_profile(a, mem_gb, gpu):
    p = dict(B.PROFILES[a.profile])
    for k in ("depth", "seq", "ratio", "iterations", "save_every"):
        v = getattr(a, k.replace("-", "_"), None)
        if v:
            p[k] = v
            if k == "ratio":
                p.pop("iterations", None)
            if k == "iterations":
                p.pop("ratio", None)
    if a.depth and a.depth != 8 and a.profile != "tiny":
        _, _, tokens = B.horizon(p)
        p["total_batch"] = B.auto_batch(tokens, p.get("ratio") or 12)
    if a.device_batch:
        p["dbs"] = a.device_batch
    elif gpu:
        p["dbs"] = B.device_batch(mem_gb, p["seq"])
    while p["total_batch"] % (p["dbs"] * p["seq"]):
        p["dbs"] = max(1, p["dbs"] // 2)
    return p


def data_ready(data_dir, need_chars):
    """(ready, built chars): manifest's train characters cover need_chars and every shard + the val shard exist."""
    man_f = Path(data_dir) / "manifest.json"
    try:
        man = json.loads(man_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, 0
    chars = sum(s["web_chars"] + s["domain_chars"] for s in man.get("shards", []))
    files = [s["file"] for s in man.get("shards", [])] + [(man.get("val") or {}).get("file", "")]
    ok = all(f and (Path(data_dir) / f).exists() for f in files)
    return ok and chars >= need_chars * 0.98, chars


def domain_inputs(rpa, a):
    rpa = Path(rpa)
    render = Path(a.render) if a.render else rpa / "train_data" / "sft_render"
    platform = Path(a.platform) if a.platform else rpa / "platform"
    cat = Path(a.catalog) if a.catalog else (render / "catalog.jsonl" if (render / "catalog.jsonl").exists()
                                             else rpa / "data" / "catalog" / "catalog.jsonl")
    return {"render": render if (render / "val.jsonl").exists() else None,
            "platform": platform if (platform / "docs").exists() or (platform / "contracts").exists() else None,
            "catalog": cat if cat.exists() else None}


def build_data(P, profile, inputs, a, cpt, log=print):
    import pretrain_data as D
    need = B.train_chars_for(profile, cpt)
    ok, have = data_ready(P["data"], need)
    if ok:
        log(f"data: {have / 1e9:.2f}B characters of shards in place ({P['data']})")
        return have
    if a.data_from == "drive":
        raise SystemExit(f"the shards from Drive cover {have / 1e9:.2f}B characters but this run needs "
                         f"{need / 1e9:.2f}B: set PRETRAIN_DATA = \"build\" in cell 1 (builds them here)")
    log(f"data: building {need / 1e9:.2f}B characters (FineWeb-Edu streamed from Hugging Face + "
        f"{100 * a.domain_share:.0f}% domain text): measured about 1 min per 1B characters plus 2 min to prepare the "
        "domain text, longer on a slow connection. A stop keeps the finished shards.")
    for k, v in inputs.items():
        if v is None:
            log(f"  note: no {k} here; the domain share comes from the other sources")
    card = D.build(P["data"], need, shard_chars=profile.get("shard_chars", 250_000_000), domain_share=a.domain_share,
                   platform=str(inputs["platform"]) if inputs["platform"] else None,
                   render_dirs=[str(inputs["render"])] if inputs["render"] else [],
                   catalog=str(inputs["catalog"]) if inputs["catalog"] else None,
                   web_dir=a.web_dir, val_groups=a.val_groups, repeat_scale=profile.get("repeat_scale", 1), log=log)
    return card["totals"]["train_chars"]


def measured_cpt(tok_dir, data_dir, n_docs=400):
    """Characters per token of this tokenizer on the start of the first train shard (the training mix)."""
    import pickle
    import pyarrow.parquet as pq
    with open(Path(tok_dir) / "tokenizer.pkl", "rb") as f:
        enc = pickle.load(f)
    pf = pq.ParquetFile(str(Path(data_dir) / "shard_00000.parquet"))
    texts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()[:n_docs]
    return sum(map(len, texts)) / max(1, sum(len(enc.encode_ordinary(t)) for t in texts))


def make_tokenizer(P, nano, a, inputs, built_chars, env, vocab=B.VOCAB, log=print):
    tok = P["tok"]
    if (tok / "tokenizer.pkl").exists() and (tok / "token_bytes.pt").exists() and B.read_card(tok):
        log(f"tokenizer: in place ({B.read_card(tok)['name']}, sha1 {B.tokenizer_sha1(tok)[:12]})")
        return
    shutil.rmtree(tok, ignore_errors=True)
    renders = [str(inputs["render"])] if inputs["render"] else []
    if a.tokenizer:
        src = Path(a.tokenizer)
        if not ((src / "tokenizer.pkl").exists() and (src / "token_bytes.pt").exists()):
            raise SystemExit(f"--tokenizer {src}: no tokenizer.pkl + token_bytes.pt there")
        tok.mkdir(parents=True)
        for f in ("tokenizer.pkl", "token_bytes.pt"):
            shutil.copy2(src / f, tok / f)
        log(f"tokenizer: {src} (Drive's v6 tokenizer: the PC and Colab share it)")
        B.tokenizer_card(tok, renders, name="v6", log=log, extra={"from": "drive"})
    else:
        chars = int(min(B.TOK_MAX_CHARS, built_chars))
        log(f"tokenizer: training {vocab:,} tokens on {chars / 1e9:.2f}B characters of the shards (a few minutes)")
        r = subprocess.run([sys.executable, "-m", "scripts.tok_train", f"--max-chars={chars}",
                            f"--vocab-size={vocab}"], cwd=str(nano), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        for line in (r.stdout + r.stderr).splitlines():          # rustbpe prints a line per 1% of the merges
            if "Progress:" not in line:
                log("   " + line)
        rc = r.returncode
        if rc != 0 or not (tok / "tokenizer.pkl").exists():
            raise SystemExit("tok_train failed (see the lines above); run again to retry")
        B.tokenizer_card(tok, renders, name="v6", log=log, extra={"trained_chars": chars})
    card = B.read_card(tok)
    if not card or not card.get("specials_ok") or card.get("vocab_size") != vocab:
        raise SystemExit(f"the tokenizer in {tok} is not usable for the chat format: {card}")


def healthy_resume_step(ckpt, log=print):
    """The newest checkpoint that is complete and all finite; incomplete / broken newer ones are removed (base_train
    saves them only through the NaN guard, so a broken one means an older tool or a cut save)."""
    for step in reversed(B.checkpoint_steps(ckpt)):
        if not B.complete_checkpoint(ckpt, step):
            log(f"  checkpoint {step} is incomplete (a stop while it was saved): removed")
            B.drop_checkpoint(ckpt, step)
            continue
        st, bad = model_health(ckpt, step)
        if st == "broken":
            log(f"  checkpoint {step} has nan/inf in {T.short_list(bad, 3)}: removed")
            B.drop_checkpoint(ckpt, step)
            continue
        if st == "unchecked" and bad and bad[0].startswith("could not read"):
            log(f"  checkpoint {step} cannot be read (cut while it was saved): removed")
            B.drop_checkpoint(ckpt, step)
            continue
        try:                                     # the optimizer state is saved last: a cut save leaves it partial
            obad = T.nonfinite_tensors(Path(ckpt) / f"optim_{step:06d}_rank0.pt")
        except ImportError:
            obad = []
        except Exception:                                   # noqa: BLE001
            obad = ["(cannot be read)"]
        if obad:
            log(f"  checkpoint {step}: its optimizer state is broken or cut ({T.short_list(obad, 3)}): removed")
            B.drop_checkpoint(ckpt, step)
            continue
        return step
    for f in Path(ckpt).glob("optim_*") if Path(ckpt).exists() else []:     # optimizer state without its model
        if not (Path(ckpt) / f.name.replace("optim_", "model_").replace("_rank0.pt", ".pt")).exists():
            f.unlink()
    return None


def follow(p, lf, total, log_every=25, watch=None, stop_after=0):
    """The run's output to the log; a progress line every log_every steps (and the first 3), validation and NaN
    lines here. stop_after: end the run once a checkpoint at or past that step is saved (tests: stop + resume).
    -> (exit code, tokens/s seen)."""
    watch = {} if watch is None else watch
    tps = []
    for line in p.stdout:
        lf.write(line)
        lf.flush()
        m = B.STEP_RE.search(line)
        v = B.VAL_RE.search(line)
        if m:
            s, dt, tp = int(m.group(1)), float(m.group(5)) / 1000, int(m.group(6).replace(",", ""))
            tps.append(tp)
            if s % log_every == 0 or s <= 2 or nan_value(m.group(4)):
                left = (total - s) * dt / 3600
                print(f"   step {s}/{total} | loss {m.group(4)} | {tp:,} tokens/s | {dt:.1f} s per step | about "
                      f"{left:.1f} h to go ({time.strftime('%H:%M')})", flush=True)
        elif v:
            print("   " + line.rstrip(), flush=True)
        elif stop_after and (m3 := re.search(r"Saved optimizer state to: .*optim_(\d+)_rank0", line)) \
                and int(m3.group(1)) >= stop_after:
            print(f"   (stopping after the checkpoint of step {int(m3.group(1))}, --stop-after)", flush=True)
            watch["stopped"] = int(m3.group(1))
            p.terminate()
        elif B.NAN_MARK in line:
            print("   " + line.rstrip(), flush=True)
            watch["nan"] = line.split(":", 1)[-1].strip()
        elif any(k in line for k in ("Traceback", "Error", "Resuming", "Vocab size", "Total number of training tokens",
                                     "Peak memory", "Saved model", "Tokens : Scaling", "out of memory",
                                     "Tokenizer sha1", "COMPUTE_DTYPE", "gradient accumulation")):
            print("   " + line.rstrip(), flush=True)
            m2 = re.search(r"Peak memory usage: ([\d.]+)MiB", line)
            if m2:
                watch["peak_mib"] = float(m2.group(1))
    return p.wait(), tps


def run_train(cmd, cwd, env, log, total, watch, stop_after=0):
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        job = windows_kill_with_parent(p)                       # noqa: F841  (kept open while it runs)
        pid_f = Path(log).with_name("child.pid")
        pid_f.write_text(str(p.pid), encoding="utf-8")
        try:
            return follow(p, lf, total, watch=watch, stop_after=stop_after)
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=60)
                except Exception:                               # noqa: BLE001
                    p.kill()
            pid_f.unlink(missing_ok=True)


def kill_leftover(run):
    pid_f = Path(run) / "child.pid"
    if not pid_f.exists():
        return
    try:
        import psutil
        pr = psutil.Process(int(pid_f.read_text(encoding="utf-8")))
        if "base_train_ooa" in " ".join(pr.cmdline()):
            print("stopping a pretraining process left running by an earlier stop, pid", pr.pid)
            pr.kill()
    except Exception:                                           # noqa: BLE001
        pass
    pid_f.unlink(missing_ok=True)


def show_plan(profile, gpu_name, mem, gpu, P, inputs, a, state):
    _, it, tokens = B.horizon(profile)
    est = B.estimate(profile, "t4" if "T4" in (gpu_name or "") else "rtx4060")
    print("device           :", gpu_name, f"({mem:.0f} GB)" if gpu else "", "| torch.compile",
          "off" if a.no_compile else "on")
    print(f"model            : d{profile['depth']} ({est['params_M']}M parameters, vocab {profile.get('vocab', B.VOCAB):,}), window "
          f"{profile['seq']}, full attention")
    print(f"horizon          : {tokens / 1e9:.2f}B tokens = {it} steps x {profile['total_batch']:,} tokens "
          f"({est['tokens_per_param']} tokens per parameter); {profile['dbs']} rows x {profile['seq']} per pass")
    print(f"estimate         : about {est['hours_range'][0]}-{est['hours_range'][1]} h on an RTX 4060-class GPU "
          f"(~{est['tokens_per_s']:,} tokens/s; --measure 20 gives this PC's real speed); saved every "
          f"{profile['save_every']} steps (about {profile['save_every'] * est['seconds_per_step'] / 60:.0f} min)")
    print(f"disk             : about {est['disk_gb']} GB (data {est['data_gb']} GB, checkpoints 2 x "
          f"{est['checkpoint_gb']} GB, the pack {est['checkpoint_gb']} GB); free here: "
          f"{shutil.disk_usage(P['run'].parent if P['run'].parent.exists() else Path(a.rpa)).free / 2**30:.0f} GB")
    print("domain text      :", ", ".join(f"{k} {v or 'MISSING'}" for k, v in inputs.items()))
    done = B.checkpoint_steps(P["ckpt"])
    print("progress         :", f"checkpoint step {done[-1]} of {it}" if done else "not started",
          "| data", "in place" if data_ready(P["data"], 1)[0] else "to build",
          "| tokenizer", "in place" if B.read_card(P["tok"]) else ("from " + a.tokenizer if a.tokenizer else "to train"))
    if state.get("finished"):
        print("finished         :", state["finished"])
    return est


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:                                           # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(RPA_DEFAULT))
    ap.add_argument("--profile", choices=sorted(B.PROFILES), default="pc")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--seq", type=int, default=0, choices=[0, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--ratio", type=float, default=0, help="tokens per scaling parameter (pc profile 45 = 1.51B tokens)")
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--device-batch", type=int, default=0, help="rows per pass (0: from GPU memory)")
    ap.add_argument("--render", default=None, help="rendered rows for the domain text (default train_data/sft_render)")
    ap.add_argument("--platform", default=None, help="platform checkout (default RPA/platform)")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--web-dir", default=None, help="local FineWeb-Edu parquet files instead of streaming (tests)")
    ap.add_argument("--val-groups", type=int, default=2)
    ap.add_argument("--domain-share", type=float, default=0.15)
    ap.add_argument("--data-from", choices=["build", "drive"], default="build",
                    help="build the shards here (default) or use the ones data_pc.py downloaded from Drive")
    ap.add_argument("--tokenizer", default=None, help="a v6 tokenizer folder to use instead of training one (Drive's)")
    ap.add_argument("--no-compile", action="store_true", default=(os.name == "nt"))
    ap.add_argument("--compile", dest="no_compile", action="store_false")
    ap.add_argument("--allow-cpu", action="store_true", help="run on a CPU (only for the tiny pipeline test)")
    ap.add_argument("--measure", type=int, default=0, help="train this many steps, print the speed, keep nothing")
    ap.add_argument("--stop-after", type=int, default=0, help="stop once this step is saved (tests: stop + resume)")
    ap.add_argument("--restart", action="store_true", help="throw the run in progress away (data, tokenizer stay)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYTHONUTF8", "1")
    rpa = Path(a.rpa).expanduser().resolve()
    a.rpa = str(rpa)
    nano = rpa / "nanochat"
    P = paths(rpa)
    dev, dev_name, mem, cap = device_info()
    gpu = dev == "cuda"
    state = json.loads(P["state"].read_text(encoding="utf-8")) if P["state"].exists() else {}
    if a.restart and not a.check:
        shutil.rmtree(P["ckpt"], ignore_errors=True)
        state = {}
        P["state"].unlink(missing_ok=True)
    profile = resolve_profile(a, mem, gpu)
    if state.get("profile") and B.checkpoint_steps(P["ckpt"]) and not a.measure:
        keep = {k: state["profile"].get(k) for k in ("depth", "seq", "total_batch", "ratio", "iterations")}
        if keep != {k: profile.get(k) for k in keep}:
            print(f"note: continuing the run in progress with its own settings ({keep}); --restart (notebook "
                  "RESTART_TRAINING = True) starts a new one with the new settings")
        profile = dict(state["profile"], dbs=profile["dbs"] if not a.device_batch else a.device_batch)
        while profile["total_batch"] % (profile["dbs"] * profile["seq"]):
            profile["dbs"] = max(1, profile["dbs"] // 2)
    inputs = domain_inputs(rpa, a)
    if a.tokenizer and not Path(a.tokenizer).exists():
        a.tokenizer = None
    show_plan(profile, dev_name, mem, gpu, P, inputs, a, state)

    problems = []
    if not gpu and not a.allow_cpu:
        problems.append("pretraining needs an NVIDIA GPU (about 11-14 h on an RTX 4060; months on a CPU)")
    if not (nano / "scripts" / "base_train.py").exists() or not (nano / "scripts" / "tok_train.py").exists():
        problems.append(f"nanochat is missing in {nano}: run notebook cell 2 again")
    else:
        miss = B.missing_anchors(pinned_script(nano, "scripts/base_train.py"))
        if miss:
            problems.append(f"{nano / 'scripts' / 'base_train.py'} is not the pinned version "
                            f"({NANOCHAT_COMMIT[:7]}); missing: {miss}")
    if inputs["render"] is None and inputs["platform"] is None and a.data_from == "build":
        problems.append("no domain text: neither train_data/sft_render (run cell 3) nor the platform files are here")
    try:
        import pyarrow  # noqa: F401
        import requests  # noqa: F401
        import tiktoken  # noqa: F401
        import rustbpe  # noqa: F401
    except ImportError as ex:
        problems.append(f"a package is missing ({ex.name}): run notebook cell 2 again")
    est = B.estimate(profile)
    free = shutil.disk_usage(rpa).free / 1e9
    have = sum(f.stat().st_size for f in P["data"].glob("*.parquet")) / 1e9 if P["data"].exists() else 0.0
    if free < (est["disk_gb"] - have) * 0.8 and not a.measure:
        problems.append(f"only {free:.0f} GB free on this drive; the run needs about {est['disk_gb']} GB")
    for p_ in problems:
        print("  !", p_)
    if problems:
        raise SystemExit("\nNot ready: fix the lines marked ! above.")
    if a.check:
        print("\nREADY.")
        return 0

    if os.name == "nt":                                     # keep the PC awake while this runs
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:                                   # noqa: BLE001
            pass
    P["run"].mkdir(parents=True, exist_ok=True)
    P["work"].mkdir(parents=True, exist_ok=True)
    kill_leftover(P["run"])
    if os.name != "nt":
        import signal
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    threads = os.cpu_count() or 4
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(P["work"]), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS=str(1 if gpu else threads), OOA_KEEP_CHECKPOINTS=str(B.KEEP_CHECKPOINTS),
               NANOCHAT_DTYPE=("bfloat16" if cap and cap >= (8, 0) else "float16") if gpu else "float32")

    # 1-2: data and tokenizer
    print("\n== 1. data", flush=True)
    cpt = B.CHARS_PER_TOKEN_MIX
    if B.read_card(P["tok"]) and (P["data"] / "shard_00000.parquet").exists():
        cpt = min(cpt, measured_cpt(P["tok"], P["data"]))     # the real tokenizer's chars per token on the mix
    built = build_data(P, profile, inputs, a, cpt)
    print("\n== 2. tokenizer", flush=True)
    make_tokenizer(P, nano, a, inputs, built, env, vocab=profile.get("vocab", B.VOCAB))
    real_cpt = measured_cpt(P["tok"], P["data"])
    print(f"   {real_cpt:.2f} characters per token on the training mix")
    if real_cpt < cpt * 0.98:                                 # fewer characters per token than assumed: more text
        built = build_data(P, profile, inputs, a, real_cpt)
    sha = B.tokenizer_sha1(P["tok"])
    for s in B.checkpoint_steps(P["ckpt"]):
        m = json.loads((P["ckpt"] / f"meta_{s:06d}.json").read_text(encoding="utf-8"))
        if m.get("tokenizer_sha1") not in (None, sha):
            raise SystemExit(f"the run in progress was trained with another tokenizer (sha1 {m['tokenizer_sha1'][:12]}) "
                             f"than {P['tok']} ({sha[:12]}): RESTART_TRAINING = True (--restart) starts it again")

    write_script(nano, compile_ok=not a.no_compile)
    if state.get("finished") and (P["pack"] / "base_checkpoints").exists() and not a.measure:
        print(f"\nalready pretrained on {state['finished']}: {P['pack']}. RESTART_TRAINING = True (--restart) "
              "pretrains again (the pack is then kept as base_v6_before_<date>).")
        return 0
    if a.measure:
        print(f"\n== measuring {a.measure} steps", flush=True)
        mdir = P["work"] / "base_checkpoints" / "measure"
        shutil.rmtree(mdir, ignore_errors=True)
        mp = dict(profile, iterations=a.measure, save_every=-1, eval_every=-1, sample_every=-1)
        mp.pop("ratio", None)
        cmd = [sys.executable, "-m", "scripts.base_train_ooa"] + B.base_train_args(
            mp, device_type=None if gpu else "cpu", model_tag="measure")
        watch = {}
        rc, tps = run_train(cmd, str(nano), env, P["log"], a.measure, watch)
        shutil.rmtree(mdir, ignore_errors=True)
        if rc != 0 or len(tps) < 2:
            raise SystemExit(f"the measuring run stopped (exit {rc}); see {P['log']}")
        steady = tps[2:] or tps[1:]
        tp = sum(steady) / len(steady)
        _, it, tokens = B.horizon(profile)
        print(f"\nMEASURED: {tp:,.0f} tokens/s at d{profile['depth']} / {profile['seq']} x {profile['dbs']} rows; peak "
              f"memory {watch.get('peak_mib', 0) / 1024:.1f} GB -> the full run ({tokens / 1e9:.2f}B tokens) takes "
              f"about {tokens / tp / 3600:.1f} h")
        return 0

    # 3: pretrain (resumable)
    print("\n== 3. pretrain", flush=True)
    _, it, tokens = B.horizon(profile)
    if not state.get("profile"):
        state = {"profile": profile, "started": time.strftime("%Y-%m-%d %H:%M"), "tokenizer_sha1": sha,
                 "nanochat": NANOCHAT_COMMIT}
    state["profile"]["dbs"] = profile["dbs"]
    P["state"].write_text(json.dumps(state, indent=1), encoding="utf-8")
    while True:
        resume = healthy_resume_step(P["ckpt"])
        if resume is not None and resume >= it:
            break
        print(("   resuming from step %d of %d" % (resume, it)) if resume else f"   starting: {it} steps", flush=True)
        cmd = [sys.executable, "-m", "scripts.base_train_ooa"] + B.base_train_args(
            profile, resume_step=resume, device_type=None if gpu else "cpu")
        watch = {}
        rc, _ = run_train(cmd, str(nano), env, P["log"], it, watch, a.stop_after)
        if watch.get("stopped"):
            break
        if watch.get("nan") or rc == B.NAN_EXIT:
            state.setdefault("nan_stops", []).append({"after_step": resume, "why": watch.get("nan"),
                                                      "time": time.strftime("%Y-%m-%d %H:%M")})
            P["state"].write_text(json.dumps(state, indent=1), encoding="utf-8")
            last = healthy_resume_step(P["ckpt"])
            raise SystemExit(f"\nSTOPPED: the pretraining went unstable ({watch.get('nan') or 'nan'}). Nothing broken "
                             f"was saved; the newest good checkpoint is step {last}. Running again resumes from it "
                             "(on the same data, so it may stop at the same place): send the 'step ... | loss' lines "
                             f"above and the end of {P['log']}.")
        if rc != 0:
            raise SystemExit(f"pretraining stopped with an error (exit {rc}; see the end of {P['log']}). Run again: it "
                             f"resumes from the newest checkpoint (step {healthy_resume_step(P['ckpt'])}).")
    final = healthy_resume_step(P["ckpt"])
    if final is None or final < it:
        print(f"stopped after step {final} (--stop-after): run again to resume")
        return 0

    # 4: pack
    print("\n== 4. pack", flush=True)
    info = {"profile": profile, "tokens": tokens, "steps": it, "nanochat": NANOCHAT_COMMIT,
            "data_manifest_sha1": hashlib.sha1((P["data"] / "manifest.json").read_bytes()).hexdigest(),
            "tokenizer": B.read_card(P["tok"])}
    pack = B.write_pack(P["work"], final, P["pack"], info=info, data_card=P["data"])
    state["finished"] = time.strftime("%Y-%m-%d %H:%M")
    state["pack"] = str(pack)
    P["state"].write_text(json.dumps(state, indent=1), encoding="utf-8")
    shutil.rmtree(P["ckpt"], ignore_errors=True)               # the pack holds the final model
    print(f"\nPRETRAINED: {pack} (step {final}, {tokens / 1e9:.2f}B tokens). The current base and the installed "
          "model are unchanged. Next: MODE = \"full\" with BASE = \"v6\" in cell 1 trains the OOA model on it. The "
          f"shards ({P['data']}) can be deleted to free disk; they are only needed to pretrain again.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped: run again to resume from the newest checkpoint", flush=True)
        sys.exit(130)
