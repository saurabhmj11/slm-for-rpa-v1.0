"""RL on this PC's NVIDIA GPU: Colab cell 9 (training/rl_setup.py) run locally (notebook MODE "rl").

    python rl_pc.py --rpa DIR --data DIR [--steps 300] [--check] [--restart]

Continues the newest trained model (train_pc.topup_start: Drive's newest, or this PC's own when it already grew from
it, e.g. a refine of Drive's v5 that cell 6 installed) with reinforcement learning: nanochat's scripts/chat_rl.py from the pinned commit, patched by
rl_setup.patch_chat_rl (written as scripts/chat_rl_ooa.py; the original stays untouched), on fresh requests built by
rl_setup.prepare (never in the training data, never close to a test prompt; prompts made by the server's Router.prompt
with --data's catalog and example index). The reward is the platform checker on the expanded flow (rl_setup.REWARD_DOC).
The run saves every --every steps and continues after any stop (run it again); a new start model starts a new run.
State: RPA/local_run/rl/ (state.json, pool/, base/ with chatsft_checkpoints + chatrl_checkpoints, train.log). The result
(state.json "result") is base/chatrl_checkpoints/d8: notebook cell 6 (eval_pc.py) scores it against the installed
model and installs it only if it is better.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA_DEFAULT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rl_setup as RL                                          # noqa: E402  (one source of truth with Colab cell 9)
from train_pc import (NANOCHAT_COMMIT, newest, topup_start, device_info, drive_id, data_sig,  # noqa: E402
                      windows_kill_with_parent, local_origin, start_label)

STEPS = 200                      # Colab cell 1 RL_STEPS
PROMPTS = 2400                   # Colab RL_PROMPTS
SAMPLES = 8
EXAMPLES_PER_STEP = 8
MAX_NEW = 1536
EVERY = 25
LR_FRAC = 0.02


def pinned_chat_rl(nano):
    """scripts/chat_rl.py of the pinned commit: from git when nanochat is a clone, else the file as it is (the zip
    ships the pinned, unpatched version)."""
    nano = Path(nano)
    if (nano / ".git").exists():
        try:
            r = subprocess.run(["git", "-C", str(nano), "show", f"{NANOCHAT_COMMIT}:scripts/chat_rl.py"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0 and r.stdout:
                return r.stdout.replace("\r\n", "\n")
        except OSError:
            pass
    return (nano / "scripts" / "chat_rl.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def write_script(nano):
    s = pinned_chat_rl(nano)
    miss = RL.missing_anchors(s)
    if miss:
        raise SystemExit(f"chat_rl.py does not match the pinned nanochat ({NANOCHAT_COMMIT[:7]}); missing: {miss}")
    out = Path(nano) / "scripts" / "chat_rl_ooa.py"
    out.write_text(RL.patch_chat_rl(s), encoding="utf-8")
    shutil.copy2(HERE.parent / "ooa_rl_task.py", Path(nano) / "tasks" / "ooa_rl.py")
    return out


def kill_leftover(run_dir):
    """An RL process left running by an earlier stop (a stopped notebook cell or a killed parent on Linux / Windows)."""
    pid_f = Path(run_dir) / "child.pid"
    if not pid_f.exists():
        return
    try:
        import psutil
        pr = psutil.Process(int(pid_f.read_text(encoding="utf-8")))
        if "chat_rl_ooa" in " ".join(pr.cmdline()):
            print("stopping an RL process left running by an earlier stop, pid", pr.pid)
            pr.kill()
    except Exception:
        pass
    pid_f.unlink(missing_ok=True)


def device_batch(mem_gb):
    """Answers per training pass at the 4096 window: the T4 (15 GB) runs 2, as SFT."""
    return 8 if mem_gb >= 60 else 4 if mem_gb >= 30 else 2 if mem_gb >= 14 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(RPA_DEFAULT), help='the "rpa model" folder')
    ap.add_argument("--data", required=True, help="the rendered data the start model trained on (sft_render)")
    ap.add_argument("--start", default=None, help="a model folder with chatsft_checkpoints/d8 (+ tokenizer); default: "
                                                  "the newest trained model (as a top-up)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--prompts", type=int, default=PROMPTS)
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--examples-per-step", type=int, default=EXAMPLES_PER_STEP)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--lr-frac", type=float, default=LR_FRAC)
    ap.add_argument("--device-batch", type=int, default=0, help="answers per training pass (0: from GPU memory)")
    ap.add_argument("--seed", type=int, default=RL.SEED)
    ap.add_argument("--allow-cpu", action="store_true", help="run on a CPU (only for testing: very slow)")
    ap.add_argument("--restart", action="store_true", help="throw away the RL run in progress and start a new one")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    rpa = Path(a.rpa).resolve()
    data = Path(a.data).resolve()
    nano, platform = rpa / "nanochat", rpa / "platform"
    schema = platform / "contracts" / "flow-schema" / "flow.v1.json"
    run_dir = rpa / "local_run" / "rl"
    state_f = run_dir / "state.json"
    if a.restart and run_dir.exists() and not a.check:
        shutil.rmtree(run_dir, ignore_errors=True)
    state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    if state.get("finished"):                 # tested by cell 6 already: the next run starts fresh
        state = {}
    # always the newest model now (never the start of an older RL run in progress: after a refine was installed that
    # would be Drive's v5); the same start model and data continue the run in progress, a new one starts a new run
    src_root = Path(a.start).resolve() if a.start else topup_start(rpa)
    src_step = newest(src_root / "chatsft_checkpoints" / "d8")
    if state.get("start") and (state.get("start"), state.get("start_step")) != (str(src_root), src_step):
        print(f"note: a newer model than the RL run in progress started from ({state['start']} step "
              f"{state.get('start_step')}): a new RL run starts from it; the earlier result is kept")
    tok = next((p for p in (src_root / "tokenizer", rpa / "nanochat_base" / "tokenizer")
                if (p / "tokenizer.pkl").exists() and (p / "token_bytes.pt").exists()), None)
    dev, name, mem, cap = device_info()
    gpu = dev == "cuda"
    problems = []
    if src_step is None:
        problems.append(f"no trained model (model_<N>.pt + meta_<N>.json) in {src_root / 'chatsft_checkpoints' / 'd8'}")
    if tok is None:
        problems.append(f"no tokenizer (tokenizer.pkl + token_bytes.pt) in {src_root / 'tokenizer'}")
    for f in (data / "examples.jsonl", data / "catalog.jsonl", data / "render_report.json"):
        if not f.exists():
            problems.append(f"missing {f}: the rendered data the model trained on (notebook cell 3)")
    if not schema.exists():
        problems.append(f"missing {schema}: the platform files (notebook cell 2)")
    if not (nano / "scripts" / "chat_rl.py").exists():
        problems.append(f"nanochat is missing in {nano}: run notebook cell 2 again")
    if not gpu and not a.allow_cpu:
        problems.append("RL needs an NVIDIA GPU (each step writes 64 answers; on a CPU about an hour per step)")
    print(f"RL on {name}{f' ({mem:.0f} GB)' if gpu else ''} | start model {src_root} step {src_step} | data {data}")
    print("  start model:", start_label(rpa, src_root))
    print(f"  {a.steps} steps x {a.examples_per_step} requests x {a.samples} answers, at most {a.max_new} new tokens, "
          f"saved every {a.every} steps")
    for p in problems:
        print("  !", p)
    if a.check or problems:
        return 1 if problems else 0

    run_dir.mkdir(parents=True, exist_ok=True)
    kill_leftover(run_dir)
    base = run_dir / "base"
    key = {"start": str(src_root), "start_step": src_step, "data": data_sig(data), "seed": a.seed}
    if {k: state.get(k) for k in key} != key:          # a new start model or new data: a new run
        if newest(base / "chatrl_checkpoints" / "d8") is not None:     # an earlier RL result is kept, never deleted
            shutil.move(str(base), str(run_dir / ("base_before_" + time.strftime("%Y%m%d_%H%M%S"))))
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(run_dir / "pool", ignore_errors=True)
        # the Drive model it grew from: an RL result installed later must keep it, or the next top-up / refine / RL
        # would start from Drive's model again instead of from it (train_pc.topup_start)
        state = dict(key, mode="rl", steps=a.steps, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                     drive_origin=drive_id(src_root) or (local_origin(rpa) if src_root == rpa / "nanochat_base" else None))
    d8 = base / "chatsft_checkpoints" / "d8"
    if newest(d8) != src_step:
        d8.mkdir(parents=True, exist_ok=True)
        for f in (f"model_{src_step:06d}.pt", f"meta_{src_step:06d}.json"):
            shutil.copy2(src_root / "chatsft_checkpoints" / "d8" / f, d8 / f)
    shutil.copytree(tok, base / "tokenizer", dirs_exist_ok=True)
    pool = run_dir / "pool"
    if not (pool / "rl_train.jsonl").exists():
        train_files = [str(p) for p in data.glob("*.jsonl") if p.name not in ("catalog.jsonl", "examples.jsonl")]
        rep = RL.prepare(str(pool), str(data / "catalog.jsonl"), str(schema), str(platform), str(data / "examples.jsonl"),
                         a.prompts, train_files=train_files, seed=a.seed, eval_dir=str(rpa / "eval"))
        print("RL requests:", rep["train"], "train /", rep["val"], "held out | by source:", rep["sources"])
    n_val = sum(1 for _ in open(pool / "rl_val.jsonl", encoding="utf-8"))
    script = write_script(nano)
    dtype = ("bfloat16" if cap and cap >= (8, 0) else "float16") if gpu else "float32"
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base), OOA_DIR=str(HERE.parent), OOA_CATALOG=str(data / "catalog.jsonl"),
               OOA_SCHEMA=str(schema), NANOCHAT_DTYPE=dtype, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    args = RL.rl_args((pool / "rl_train.jsonl").as_posix(), (pool / "rl_val.jsonl").as_posix(), a.steps,
                      samples=a.samples, examples_per_step=a.examples_per_step,
                      device_batch=a.device_batch or (device_batch(mem) if gpu else 1), max_new=a.max_new,
                      every=a.every, eval_examples=n_val, init_lr_frac=a.lr_frac, device_type="" if gpu else "cpu")
    if a.samples % int([x for x in args if x.startswith("--device-batch-size=")][0].split("=")[1]):
        raise SystemExit("--samples must be a multiple of the device batch")
    cmd = [sys.executable, "-u", "-m", f"scripts.{script.stem}"] + args
    state.update(result=str(base / "chatrl_checkpoints" / "d8"), tokenizer=str(base / "tokenizer"), data=str(data))
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    log = run_dir / "train.log"
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=nano, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        job = windows_kill_with_parent(p)
        pid_f = run_dir / "child.pid"
        pid_f.write_text(str(p.pid), encoding="utf-8")
        try:
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                if " INFO " not in line and "Scaling the LR" not in line:
                    print(line, end="", flush=True)
            rc = p.wait()
        finally:
            if p.poll() is None:            # a stopped cell must not leave RL running
                p.terminate()
                try:
                    p.wait(timeout=30)
                except Exception:
                    p.kill()
            pid_f.unlink(missing_ok=True)
            if job:
                try:
                    import ctypes
                    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(job))
                except Exception:
                    pass
    rl_step = newest(base / "chatrl_checkpoints" / "d8")
    state.update(rl_step=rl_step, done=rc == 0 and rl_step == a.steps - 1,
                 install_step=(src_step or 0) + (rl_step or 0) + 1)
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"\nRL {'finished' if state['done'] else 'stopped'} at step {rl_step} | result {state['result']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
