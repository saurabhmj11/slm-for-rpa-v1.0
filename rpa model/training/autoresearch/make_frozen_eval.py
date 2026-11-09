"""Build the FROZEN scoreboard once: unseen generated requests + held-out real flows, plus a snapshot of the catalog they were
checked against. Never regenerate it after experiments start (the hash would change and results would stop being comparable).
Usage: python make_frozen_eval.py PLATFORM_REPO CATALOG.jsonl OUT_DIR"""
import hashlib, json, os, sys, glob, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gen_flows as G

repo, catalog_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
cat = [json.loads(l) for l in open(catalog_path, encoding="utf-8")]
rows = []
for q, f, mini in G.generate(cat, repo, 40, 777):
    rows.append({"kind": "synthetic", "prompt": q, "gold_flow": mini,
                 "gold_actions": [s["action"] for s in G.generate.__globals__["json"].loads(mini)["steps"]]})
# gold_actions must include nested steps
def acts(flow):
    res = []
    def walk(ns):
        for n in ns or []:
            res.append(n["action"])
            for k in ("children", "else", "finally"):
                if isinstance(n.get(k), list): walk(n[k])
            for c in n.get("catch") or []: walk(c.get("children"))
    walk(flow.get("steps")); return res
for r in rows:
    r["gold_actions"] = acts(json.loads(r["gold_flow"]))
import hashlib as _h
for path in sorted(glob.glob(repo + "/examples/**/*.flow.json", recursive=True)):
    fam = re.sub(r"-capture$", "", os.path.basename(path)[:-len(".flow.json")])
    if int(_h.sha1(fam.encode()).hexdigest(), 16) % 4 != 0 or path.endswith("-capture.flow.json"):
        continue
    d = json.load(open(path, encoding="utf-8"))
    if (d.get("description") or "").strip() and d.get("steps"):
        rows.append({"kind": "real", "prompt": d["description"].strip(),
                     "gold_flow": json.dumps(d, separators=(",", ":"), ensure_ascii=False), "gold_actions": acts(d)})
body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
(out / "frozen_eval.jsonl").write_text(body, encoding="utf-8")
(out / "frozen_catalog.jsonl").write_text(open(catalog_path, encoding="utf-8").read(), encoding="utf-8")
sha = hashlib.sha256(body.encode()).hexdigest()
(out / "frozen_eval.sha256").write_text(sha + "\n")
print(f"frozen eval: {sum(r['kind']=='synthetic' for r in rows)} synthetic + {sum(r['kind']=='real' for r in rows)} real | sha256 {sha[:16]}...")
