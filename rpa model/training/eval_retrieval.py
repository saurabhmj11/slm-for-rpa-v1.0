"""Measure action retrieval on REAL flows: human-written flow description -> are the flow's real actions in the top-k?
Usage: python eval_retrieval.py PLATFORM_REPO     (no GPU, standard library only)
Result on the 2026-11-07 platform snapshot: top-10 finds ~10% of the actions real flows use; none of 31 flows has all of
them. Real flows average 18.5 steps, 11 distinct actions, 3.7 toolkits. Generated flows (1-4 steps) are far easier."""
import glob, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval as R, importlib.util
spec = importlib.util.spec_from_file_location("bc", Path(__file__).resolve().parent.parent / "data/catalog/build_catalog.py")
bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
repo = sys.argv[1]
cat = []
for p in sorted(Path(repo, "docs/reference/toolkits").glob("*.md")):
    if p.name != "README.md":
        cat += bc.parse_file(p)
idx = R.ActionIndex(cat)

def acts(flow):
    out = []
    def walk(ns):
        for n in ns or []:
            out.append(n["action"])
            for k in ("children", "else", "finally"):
                walk(n.get(k) if isinstance(n.get(k), list) else [])
            for c in n.get("catch") or []:
                walk(c.get("children"))
    walk(flow.get("steps")); return [a for a in out if a in idx.by_id]

rows = []
for f in sorted(glob.glob(repo + "/examples/**/*.flow.json", recursive=True)):
    d = json.load(open(f)); a = acts(d); desc = (d.get("description") or "").strip()
    if desc and a and not f.endswith("-capture.flow.json"):
        rows.append((desc, set(a)))
print(f"{len(rows)} real flows, {len(cat)} catalog entries")
for k in (5, 10, 20, 40):
    tot = hit = full = 0
    for desc, a in rows:
        top = {e["id"] for e in idx.search(desc, k)}
        tot += len(a); hit += len(a & top); full += a <= top
    print(f"top-{k}: {hit}/{tot} actions found ({100 * hit / tot:.0f}%), flows with all actions found: {full}/{len(rows)}")
