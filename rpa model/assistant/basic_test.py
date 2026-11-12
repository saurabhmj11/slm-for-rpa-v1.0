"""Send the 20 everyday test requests (eval/plain_requests.jsonl) to the RUNNING assistant and print a score.

Start the assistant first (start_assistant.bat), then in a second Command Prompt:
    python basic_test.py                 (assistant on the default port 8765)
    python basic_test.py --port 8799
Standard library only. Takes about 2-5 minutes on a CPU. Writes basic_test_results.json next to this file.

Per request it reports: verified (passed the platform's rules) and right (the flow also contains the actions the
request needs, in that order; see "must" in plain_requests.jsonl).
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALS = [HERE.parent / "eval" / "plain_requests.jsonl",      # set 1: the 20 everyday requests
         HERE.parent / "eval" / "plain_requests_2.jsonl"]    # set 2: 15 more, never used to pick a model by hand


def actions(steps):
    out = []
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        if isinstance(s.get("action"), str):
            out.append(s["action"])
        for k in ("children", "else", "finally"):
            out += actions(s.get(k))
        for c in s.get("catch") or []:
            if isinstance(c, dict):
                out += actions(c.get("children"))
    return out


def in_order(must, got):
    it = iter(got)
    return all(any(a == m for a in it) for m in must)


def build(port, request):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/build", json.dumps({"request": request}).encode("utf-8"),
                                 {"X-OOA-Assistant": "1", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    rows = [dict(json.loads(l), set=i) for i, p in enumerate(EVALS, 1) if p.exists()
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    try:
        model = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{a.port}/api/status", headers={"X-OOA-Assistant": "1"}), timeout=10)).get("model")
    except Exception as ex:
        raise SystemExit(f"The assistant is not running on port {a.port} ({ex}). Start start_assistant.bat first.")
    print("model:", model)
    results, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        d = build(a.port, r["prompt"])
        got = actions((d.get("flow") or {}).get("steps"))
        ok = bool(d.get("verified"))
        right = ok and in_order(r["must"], got)
        results.append({"request": r["prompt"], "set": r["set"], "verified": ok, "right": right, "actions": got,
                        "problems": d.get("problems", [])[:3], "seconds": d.get("seconds")})
        mark = "RIGHT   " if right else "verified" if ok else "failed  "
        print(f"{i:2}. {mark} {d.get('seconds', 0):6.1f}s  {r['prompt'][:60]}")
    for i in sorted({x["set"] for x in results}):
        d = [x for x in results if x["set"] == i]
        print(f"\nset {i}: verified {sum(x['verified'] for x in d)}/{len(d)} | right (needed actions in order) "
              f"{sum(x['right'] for x in d)}/{len(d)}", end="")
    v, ok = sum(x["verified"] for x in results), sum(x["right"] for x in results)
    print(f"\nall   : verified {v}/{len(rows)} | right {ok}/{len(rows)} | {time.time() - t0:.0f} s")
    (HERE / "basic_test_results.json").write_text(json.dumps({"model": model, "results": results}, indent=1),
                                                  encoding="utf-8")
    print("details saved to", HERE / "basic_test_results.json")


if __name__ == "__main__":
    main()
