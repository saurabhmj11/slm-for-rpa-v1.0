"""Long training on a PC without a GPU: the Colab recipe (2500 steps from the pretrained base model), split into
chunks that are each saved, so a restart, update or sleep loses at most one chunk. Run it again to continue.

    python train_long.py                 start, or continue where it stopped (a finished run starts a new one)
    python train_long.py --check         show what would be used and the time estimate
    python train_long.py --restart       throw away the saved chunks and start over

Each chunk (default 250 steps, about 2.5 h on 4 CPU cores) starts from the previous chunk's model and optimizer, sees
a different shuffle of the training data, and uses a slightly lower learning rate than the one before. When all
chunks are done the result is scored against the assistant's current model on the 20 everyday requests and installed
only if it is better (the old model is kept in nanochat_base/old_model_<date>).
Needs in "rpa model": nanochat_base/base_checkpoints/d8 (pretrained base model), nanochat_base/tokenizer, sft/ with
the training data, nanochat/, platform/, assistant/. The notebook ooa_local_pc.ipynb (MODE = "long") downloads them.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topup_local as T                                         # newest(), ident(), read_rows(), score()

REQUIRED = ["train_flows.jsonl", "train_catalog.jsonl", "synth_train.jsonl", "synth_pattern.jsonl", "val.jsonl"]


def keep_awake():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def copy_model(src_dir, step, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if re.match(rf"(model|meta|optim)_{step:06d}", f.name):
            shutil.copy2(f, dst_dir / f.name)


def main():
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        pass
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(HERE.parent.parent), help='the "rpa model" folder of the assistant')
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--chunk-steps", type=int, default=250, help="steps per chunk (32,768 tokens each); 10 x 250 = Colab's 2500")
    ap.add_argument("--device-batch", type=int, default=2, choices=[1, 2, 4, 8, 16],
                    help="rows in memory at once: 2 uses about 4-5 GB RAM (4: about 7 GB, 1: about 3 GB); same result")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="CPU threads (default: all but one, so the PC stays usable)")
    ap.add_argument("--full-speed", action="store_true", help="normal priority and all threads (e.g. overnight)")
    ap.add_argument("--smoltalk", type=int, default=300, help="general chat rows per chunk (downloaded once)")
    ap.add_argument("--feedback-repeat", type=int, default=5)
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--restart", action="store_true", help="start over (the saved chunks are deleted)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    rpa = Path(a.rpa).expanduser().resolve()
    base, sft = rpa / "nanochat_base", rpa / "sft"
    run_dir = rpa / "long_run"
    state_f = run_dir / "state.json"
    bd8 = base / "base_checkpoints" / "d8"
    bstep = T.newest(bd8)
    catalog = sft / "catalog.jsonl" if (sft / "catalog.jsonl").exists() else rpa / "data" / "catalog" / "catalog.jsonl"
    schema = rpa / "platform" / "contracts" / "flow-schema" / "flow.v1.json"
    fb = rpa / "assistant" / "assistant_data" / "feedback.jsonl"

    problems = []
    if bstep is None:
        problems.append(f"no pretrained base model in {bd8} (model_<N>.pt + meta_<N>.json + optim_<N>_rank0.pt)")
    for f in [base / "tokenizer" / "tokenizer.pkl", base / "tokenizer" / "token_bytes.pt", schema,
              rpa / "nanochat" / "scripts" / "chat_sft.py", rpa / "assistant" / "app.py"] + [sft / r for r in REQUIRED]:
        if not f.exists():
            problems.append(f"missing {f}")
    if a.restart and run_dir.exists() and not a.check:
        shutil.rmtree(run_dir, ignore_errors=True)
    state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    if state and (state.get("chunks") != a.chunks or state.get("chunk_steps") != a.chunk_steps):
        problems.append(f"a run with {state.get('chunks')} x {state.get('chunk_steps')} steps is in progress: use the same "
                        "--chunks/--chunk-steps to continue it, or --restart")
    if state.get("finished"):                                   # the last run is complete: this starts a new one
        print(f"The last long run finished on {state['finished']} (results in nanochat_base/topup_history.jsonl): "
              "starting a new one.")
        if not a.check:
            shutil.rmtree(run_dir, ignore_errors=True)
        state = {}
    done = state.get("done", 0)
    left = a.chunks - done
    per_chunk_h = a.chunk_steps * 38 / 3600 * 4 / max(1, min(os.cpu_count() or 4, 16))
    print("assistant folder :", rpa)
    print("base model       :", f"{bd8} step {bstep}" if bstep is not None else "-")
    print(f"plan             : {a.chunks} chunks x {a.chunk_steps} steps ({done} done, {left} to go)")
    print(f"time estimate    : about {per_chunk_h:.1f} h per chunk on this PC's {os.cpu_count()} CPU threads, "
          f"about {per_chunk_h * left:.0f} h in total (rough: real speed depends on the CPU)")
    for p_ in problems:
        print("  !", p_)
    if problems:
        raise SystemExit("\nNot ready: fix the lines marked ! above.")
    if a.check:
        print("\nREADY.")
        return

    keep_awake()
    run_dir.mkdir(exist_ok=True)
    state.update(chunks=a.chunks, chunk_steps=a.chunk_steps, base_step=bstep, done=done)
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    log = run_dir / "train_long.log"
    fb_rows = []
    if fb.exists():
        r = subprocess.run([sys.executable, str(rpa / "training" / "ingest_feedback.py"), str(fb), str(run_dir / "fb"),
                            str(catalog), str(schema)], capture_output=True, text=True)
        print("feedback:", r.stdout.strip() or r.stderr.strip()[-300:])
        if (run_dir / "fb" / "feedback_train.jsonl").exists():
            fb_rows = T.read_rows(run_dir / "fb" / "feedback_train.jsonl")

    for k in range(done, a.chunks):
        t0 = time.time()
        lr = 0.8 - (0.8 - 0.25) * (k / max(1, a.chunks - 1))     # steps down from Colab's 0.8 to 0.25
        if k == 0:
            src_dir, src_step = bd8, bstep                       # the pretrained base model, like Colab
        else:
            src_dir = run_dir / "chunks" / f"chunk_{k - 1:02d}"
            src_step = T.newest(src_dir)
        print(f"\n== chunk {k + 1}/{a.chunks}: {a.chunk_steps} steps, learning rate {lr:.2f}, "
              f"from {'base model' if k == 0 else f'chunk {k}'} ({time.strftime('%H:%M')})")
        work = run_dir / "work"
        shutil.rmtree(work, ignore_errors=True)
        wb = work / "nanochat_base"
        copy_model(src_dir, src_step, wb / "base_checkpoints" / "d8")
        shutil.copytree(base / "tokenizer", wb / "tokenizer")
        if (run_dir / "task_data").exists():                     # general chat data downloaded by an earlier chunk
            shutil.copytree(run_dir / "task_data", wb / "task_data")
        ws = work / "sft"
        ws.mkdir()
        rng = random.Random(f"chunk{k}")
        for f in sft.glob("*.jsonl"):
            if f.name in ("catalog.jsonl", "feedback_train.jsonl"):
                continue
            rows = T.read_rows(f)
            if f.name != "val.jsonl":
                rng.shuffle(rows)                                # every chunk meets the data in a different order
            (ws / f.name).write_text("".join(rows), encoding="utf-8")
        if fb_rows:
            (ws / "feedback_train.jsonl").write_text("".join(fb_rows * a.feedback_repeat), encoding="utf-8")
        cmd = [sys.executable, str(HERE / "train_local.py"), "--work", str(work), "--nanochat", str(rpa / "nanochat"),
               "--steps", str(a.chunk_steps), "--device-batch", str(a.device_batch), "--init-lr-frac", f"{lr:.3f}",
               "--smoltalk", str(a.smoltalk), "--eval-every", str(a.chunk_steps), "--allow-cpu"]
        if not a.full_speed:
            cmd += ["--threads", str(a.threads), "--low-priority"]
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n==== chunk {k + 1}/{a.chunks} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                m = re.match(r"step (\d+) \(([\d.]+)%\) \| loss: ([\d.]+).*?dt: ([\d.]+)ms", line)
                if m and (int(m.group(1)) % 10 == 0 or int(m.group(1)) == 1):   # a progress line every 10 steps
                    left = (a.chunk_steps - int(m.group(1))) * float(m.group(4)) / 60000
                    print(f"   step {int(m.group(1))}/{a.chunk_steps} | loss {m.group(3)} | {float(m.group(4)) / 1000:.0f} s per step"
                          f" | chunk done in about {left:.0f} min ({time.strftime('%H:%M')})")
            rc = p.wait()
        if rc != 0:
            raise SystemExit(f"chunk {k + 1} failed (see the end of {log}). Run again to retry this chunk; "
                             "the chunks before it are kept.")
        cd8 = wb / "chatsft_checkpoints" / "d8"
        cstep = T.newest(cd8)
        if cstep is None:
            raise SystemExit(f"chunk {k + 1} produced no model (see {log}). Run again to retry it.")
        if not (run_dir / "task_data").exists() and (wb / "task_data").exists():
            shutil.copytree(wb / "task_data", run_dir / "task_data")
        out = run_dir / "chunks" / f"chunk_{k:02d}"
        shutil.rmtree(out, ignore_errors=True)
        copy_model(cd8, cstep, out)
        old = run_dir / "chunks" / f"chunk_{k - 2:02d}"          # keep the last two chunks only (disk space)
        if k >= 2 and old.exists():
            shutil.rmtree(old, ignore_errors=True)
        state["done"] = k + 1
        state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
        shutil.rmtree(work, ignore_errors=True)
        h = (time.time() - t0) / 3600
        print(f"   chunk {k + 1} saved ({h:.1f} h); about {h * (a.chunks - k - 1):.0f} h to go")

    # all chunks done: score against the assistant's model and install only if better
    final_dir = run_dir / "chunks" / f"chunk_{a.chunks - 1:02d}"
    fstep = T.newest(final_dir)
    ev = run_dir / "eval_base"
    shutil.rmtree(ev, ignore_errors=True)
    copy_model(final_dir, fstep, ev / "chatsft_checkpoints" / "d8")
    shutil.copytree(base / "tokenizer", ev / "tokenizer")
    d8 = base / "chatsft_checkpoints" / "d8"
    cur = T.newest(d8) if d8.exists() else None
    print(f"\n== scoring on the {len(T.test_rows(rpa))} test requests")
    n_right, n_ver, details = T.score(rpa, ev, catalog, a.port, log)
    print(f"  new model    : {T.summary(details)}")
    if cur is not None:
        b_right, b_ver, b_details = T.score(rpa, base, catalog, a.port, log)
        print(f"  current model: {T.summary(b_details)}")
    else:
        b_right, b_ver = -1, -1
    better = (n_right, n_ver) > (b_right, b_ver)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    total = a.chunks * a.chunk_steps
    rec = {"time": stamp, "mode": "long", "trained_steps": total, "from_step": bstep,
           "old_score": [b_right, b_ver], "new_score": [n_right, n_ver], "installed": better, "details": details}
    if better:
        if cur is not None:
            keep = base / f"old_model_{stamp}"
            keep.mkdir()
            for f in d8.iterdir():
                if re.match(r"(model|meta|optim)_\d+", f.name):
                    shutil.move(str(f), keep / f.name)
            rec["old_kept_in"] = str(keep)
        d8.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_dir / f"model_{fstep:06d}.pt", d8 / f"model_{total:06d}.pt")
        m = json.loads((final_dir / f"meta_{fstep:06d}.json").read_text(encoding="utf-8"))
        m["step"] = total
        m["ooa_topup"] = {"time": stamp, "mode": "long", "from_step": bstep, "steps": total, "score": [n_right, n_ver]}
        (d8 / f"meta_{total:06d}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        rec.update(kept=f"model_{total:06d}.pt", kept_score=[n_right, n_ver], kept_id=T.ident(d8 / f"model_{total:06d}.pt"),
                   kept_details=details, tests=len(T.test_rows(rpa)))
        print(f"  BETTER: installed model_{total:06d}.pt; restart the assistant to use it")
    else:
        print("  not better than the current model: the current one stays. The trained model is kept in", final_dir)
    with open(base / "topup_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    state["finished"] = stamp
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    shutil.rmtree(ev, ignore_errors=True)
    print("long run finished. Running it again starts a new one.")


if __name__ == "__main__":
    main()
