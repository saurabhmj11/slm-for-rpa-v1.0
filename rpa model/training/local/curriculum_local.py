"""Curriculum: train the local model in small focused sessions, one module at a time, then a mixed polish.

Why: one big mixed round teaches everything at once and can trade one skill for another. Here each session trains
mostly one module (60%) plus a replay of everything else (40%, so other skills are kept), is sized to about two passes
over that module's examples (no memorising), and is scored on BOTH test sets plus that module's held-out examples
(never trained on). A session is installed only if the module improves, the test requests do not get worse, and at
most one request the model already got right is forgotten. Otherwise the best model so far stays.

Order: easy and separate first, combined last.
  notask -> files -> tables -> excel -> word -> pdf -> browser -> email -> apps -> chains -> loops -> conditions -> polish

    python curriculum_local.py                     (all stages; CPU: about 12-20 h, finished stages stay installed)
    python curriculum_local.py --stages excel,word,pdf
    python curriculum_local.py --from loops        (continue from a stage)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA_DEFAULT = HERE.parent.parent
ORDER = ["notask", "files", "tables", "excel", "word", "pdf", "browser", "email", "apps", "chains", "loops", "conditions",
         "polish"]


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
    ap.add_argument("--stages", default=",".join(ORDER), help="comma-separated stages to run, in this order")
    ap.add_argument("--from", dest="start", default=None, help="skip the stages before this one")
    ap.add_argument("--max-steps", type=int, default=120, help="largest focused session (steps); small modules get fewer")
    ap.add_argument("--polish-steps", type=int, default=150, help="final mixed session")
    ap.add_argument("--lr", type=float, default=0.2, help="learning rate of the focused sessions")
    ap.add_argument("--polish-lr", type=float, default=0.12)
    ap.add_argument("--max-forget", type=int, default=1)
    ap.add_argument("--device-batch", type=int, default=2, choices=[1, 2, 4, 8, 16])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rpa = Path(a.rpa).expanduser().resolve()
    history = rpa / "nanochat_base" / "topup_history.jsonl"
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ORDER]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {ORDER}")
    if a.start:
        stages = stages[stages.index(a.start):] if a.start in stages else stages
    base = [sys.executable, "-u", str(HERE / "topup_local.py"), "--rpa", str(rpa), "--mode", "topup",
            "--device-batch", str(a.device_batch), "--max-forget", str(a.max_forget)]
    if a.check:
        sys.exit(subprocess.run(base + ["--check"]).returncode)

    print(f"curriculum: {' -> '.join(stages)}")
    t0, results, fails = time.time(), [], 0
    for k, st in enumerate(stages, 1):
        if st == "polish":
            cmd = base + ["--steps", str(a.polish_steps), "--lr", str(a.polish_lr)]
        else:
            cmd = base + ["--focus", st, "--auto-steps", "--steps", str(a.max_steps), "--lr", str(a.lr)]
        print(f"\n######## stage {k}/{len(stages)}: {st} ########", flush=True)
        before = last_round(history)
        code = subprocess.run(cmd).returncode
        rec = last_round(history)
        if code != 0 or rec is None or (before is not None and rec.get("time") == before.get("time")):
            print(f"stage {st} did not finish (exit code {code}); the installed model is unchanged.")
            fails += 1
            if st != "polish" and code != 0 and fails < 2:
                print("  (a module with too few examples is skipped; anything else: see the log, then use --from)")
                results.append((st, "skipped"))
                continue
            print(f"stopping; continue later with --from {st}")
            sys.exit(code or 1)
        fails = 0
        sets = per_set(rec.get("kept_details"))
        s1, s2 = sets.get(1, [0, 0]), sets.get(2, [0, 0])
        ms = rec.get("module_score") or [0, 0, 0]
        note = f" | {st} held-out: {ms[0]} -> {ms[1]} of {ms[2]}" if ms[2] else ""
        state = "INSTALLED" if rec.get("installed") else "kept the previous model"
        print(f"stage {st}: {state}{note} | best model: set 1 {s1[0]}/{s1[1]}, set 2 {s2[0]}/{s2[1]} right", flush=True)
        results.append((st, state + note))
    print(f"\ncurriculum done in {(time.time() - t0) / 3600:.1f} h")
    for st, r in results:
        print(f"  {st:11s} {r}")
    print("restart the assistant to use the best model")


if __name__ == "__main__":
    main()
