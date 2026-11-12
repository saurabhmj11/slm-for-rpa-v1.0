"""Refine the local model in several top-up rounds (day and night), keeping only improvements.

Each round is one topup_local.py run from the current best model: a fresh shuffle of all the training data
(everyday wording, core actions, chains, combinations, not-a-task replies, your feedback), then scoring on both test
sets. The new model is installed only if it scores higher AND does not forget more than --max-forget requests the
current model got right, so progress is never lost. Each round uses a smaller learning rate (big fixes first, then
finer polishing). It stops when --patience rounds in a row did not improve, at the target score, or after --rounds.

    python refine_local.py                         (4 rounds of 200 steps; CPU: about 3-4 h per round)
    python refine_local.py --rounds 6 --steps 150 --target-set1 18 --target-set2 12

Rounds already finished stay installed if you stop it; run it again to continue from the best model.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA_DEFAULT = HERE.parent.parent


def last_round(history):
    if not history.exists():
        return None
    lines = [l for l in history.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def per_set(details):
    out = {}
    for d in details or []:
        s = out.setdefault(d.get("set", 1), [0, 0])
        s[0] += bool(d.get("right"))
        s[1] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(RPA_DEFAULT))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200, help="training steps per round")
    ap.add_argument("--lrs", default="0.3,0.2,0.15,0.1", help="learning rate per round (the last one repeats)")
    ap.add_argument("--patience", type=int, default=2, help="stop after this many rounds in a row without improvement")
    ap.add_argument("--max-forget", type=int, default=1)
    ap.add_argument("--target-set1", type=int, default=19, help="stop when set 1 has at least this many right ...")
    ap.add_argument("--target-set2", type=int, default=13, help="... and set 2 at least this many")
    ap.add_argument("--device-batch", type=int, default=2, choices=[1, 2, 4, 8, 16])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rpa = Path(a.rpa).expanduser().resolve()
    history = rpa / "nanochat_base" / "topup_history.jsonl"
    lrs = [float(x) for x in a.lrs.split(",") if x.strip()]
    topup = [sys.executable, "-u", str(HERE / "topup_local.py"), "--rpa", str(rpa), "--mode", "topup",
             "--steps", str(a.steps), "--device-batch", str(a.device_batch), "--max-forget", str(a.max_forget)]
    if a.check:
        sys.exit(subprocess.run(topup + ["--check"]).returncode)

    print(f"refining: up to {a.rounds} rounds of {a.steps} steps, learning rates {lrs}, stop after {a.patience} "
          f"rounds without improvement or at set 1 >= {a.target_set1} and set 2 >= {a.target_set2}")
    t0, misses, installed = time.time(), 0, 0
    for r in range(a.rounds):
        lr = lrs[min(r, len(lrs) - 1)]
        print(f"\n######## round {r + 1}/{a.rounds} | learning rate {lr} ########", flush=True)
        before = last_round(history)
        code = subprocess.run(topup + ["--lr", str(lr)]).returncode
        rec = last_round(history)
        if code != 0 or rec is None or rec is before or (before and rec.get("time") == before.get("time")):
            print(f"round {r + 1} did not finish (exit code {code}); the installed model is unchanged. Stopping.")
            sys.exit(code or 1)
        sets = per_set(rec.get("kept_details"))
        s1, s2 = sets.get(1, [0, 0]), sets.get(2, [0, 0])
        print(f"round {r + 1}: {'INSTALLED' if rec.get('installed') else 'not installed'} | best model now: "
              f"set 1 {s1[0]}/{s1[1]} right, set 2 {s2[0]}/{s2[1]} right"
              + (f" | forgot: {len(rec.get('forgot') or [])}" if rec.get("forgot") else ""), flush=True)
        if rec.get("installed"):
            installed, misses = installed + 1, 0
        else:
            misses += 1
        if s1[0] >= a.target_set1 and s2[0] >= a.target_set2:
            print("target reached: stopping")
            break
        if misses >= a.patience:
            print(f"{misses} rounds in a row without improvement: stopping (more of the same data will not help;"
                  " new kinds of examples or your feedback will)")
            break
    print(f"\nrefine done in {(time.time() - t0) / 3600:.1f} h | rounds installed: {installed} | history: {history}")
    print("restart the assistant to use the best model")


if __name__ == "__main__":
    main()
