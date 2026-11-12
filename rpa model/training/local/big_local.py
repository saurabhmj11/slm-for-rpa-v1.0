"""Big local training: clean all the training data, then train the full recipe from the pretrained base model on it,
on this PC's GPU (or CPU), for as long as it takes. Saved in chunks: a stop, restart or crash loses at most one
chunk; run it again to continue. At the end the result is scored on both test sets and installed only if better.

    python big_local.py                  filter + train (4 passes over the cleaned data), or continue a run
    python big_local.py --passes 6       more passes
    python big_local.py --check          show the plan only

1. training/data_filter.py: sft/ -> sft_clean/ (every row checked: structure, length, leaked variable names or ids,
   repeated clauses, platform rules after adding missing open/save steps, stale action ids, wrong app, too close
   to a test request, duplicates, conflicting labels, over-represented action sequences); report in
   sft_clean/filter_report.json. Done once per run (a continued run keeps its data).
2. train_long.py --sft-dir sft_clean --passes N: chunked training from the base model, learning rate stepping down.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(HERE.parent.parent))
    ap.add_argument("--passes", type=float, default=4.0)
    ap.add_argument("--chunk-steps", type=int, default=0, help="steps per saved chunk (default: 500 on a GPU, 250 on CPU)")
    ap.add_argument("--device-batch", type=int, default=2, choices=[1, 2, 4, 8, 16])
    ap.add_argument("--cap", type=int, default=60, help="data filter: most rows per action sequence")
    ap.add_argument("--full-speed", action="store_true", help="CPU: all threads and normal priority")
    ap.add_argument("--restart", action="store_true", help="throw away a run in progress and its cleaned data")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rpa = Path(a.rpa).expanduser().resolve()
    sft, clean = rpa / "sft", rpa / "sft_clean"
    state_f = rpa / "long_run" / "state.json"
    state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    in_progress = bool(state) and not state.get("finished") and not a.restart
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    chunk = a.chunk_steps or (500 if gpu else 250)
    print(f"device: {'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU (this will take days; a GPU takes hours)'}")

    if in_progress and clean.exists():
        print("continuing the run in progress with its cleaned data in", clean)
    elif not a.check:
        print("== cleaning the training data")
        r = subprocess.run([sys.executable, str(rpa / "training" / "data_filter.py"), str(sft), str(clean),
                            str(rpa / "platform"), str(rpa / "eval"), "--cap", str(a.cap)])
        if r.returncode != 0:
            sys.exit(r.returncode)
    cmd = [sys.executable, "-u", str(HERE / "train_long.py"), "--rpa", str(rpa), "--sft-dir", str(clean if clean.exists() else sft),
           "--passes", str(a.passes), "--chunk-steps", str(chunk), "--device-batch", str(a.device_batch)]
    if a.full_speed or gpu:
        cmd.append("--full-speed")
    if a.restart:
        cmd.append("--restart")
    if a.check:
        cmd.append("--check")
    print("== training:", " ".join(cmd[2:]))
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
