"""Run every check that needs no GPU, on any computer (Windows, Mac, Linux). Needs Python 3.10+ and `pip install jsonschema`.

Usage:  python local_check.py PATH_TO_OOA_PLATFORM_REPO
Writes local_check_report.txt next to this file. Nothing is installed or changed; temporary files go to a temp folder.
"""
import os, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) < 2:
    sys.exit("Usage: python local_check.py PATH_TO_OOA_PLATFORM_REPO   (the folder that contains contracts/ and docs/)")
PLATFORM = Path(sys.argv[1]).resolve()
schema = PLATFORM / "contracts" / "flow-schema" / "flow.v1.json"
if not schema.exists():
    sys.exit(f"Not the platform repo: {schema} not found.")
try:
    import jsonschema  # noqa: F401
except ImportError:
    sys.exit("Missing package. Run:  pip install jsonschema")

env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
tmp = Path(tempfile.mkdtemp())
catalog = HERE / "data" / "catalog" / "catalog.jsonl"
steps = [
    ("Build catalog from the platform docs", [str(HERE / "data/catalog/build_catalog.py"), str(PLATFORM), str(tmp / "catalog_fresh.jsonl")], "entries"),
    ("48 gold flows pass schema + catalog", [str(HERE / "eval/flow_check.py"), str(PLATFORM), str(catalog)], "unknown actions 0"),
    ("Generate 2000 validated flows", [str(HERE / "training/gen_flows.py"), str(catalog), str(PLATFORM), str(tmp / "flows.jsonl"), "2000", "1"], "2000 flows"),
    ("Verifier / reward tests (12 checks)", [str(HERE / "training/test_reward.py"), str(PLATFORM)], "all reward tests passed"),
    ("Server + router tests (8 end-to-end, fake models)", [str(HERE / "serving/test_server.py"), str(PLATFORM)], "ALL SERVER TESTS PASSED"),
    ("Autoresearch search-loop tests", [str(HERE / "training/autoresearch/test_loop.py")], "ALL LOOP TESTS PASSED"),
    ("Retrieval on real flows (expect ~10% at top-10)", [str(HERE / "training/eval_retrieval.py"), str(PLATFORM)], "top-10"),
]
rows, lines = [], []
for name, cmd, must in steps:
    t = time.time()
    p = subprocess.run([sys.executable, "-X", "utf8"] + cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    ok = p.returncode == 0 and must in out
    rows.append((name, ok, time.time() - t))
    lines.append(f"=== {name}: {'PASS' if ok else 'FAIL'}\n" + out.strip()[-1500:] + "\n")
    print(f"{'PASS' if ok else 'FAIL'}  {time.time() - t:5.1f}s  {name}")
    if not ok:
        print("      last output:", out.strip()[-400:].replace("\n", "\n      "))
passed = sum(ok for _, ok, _ in rows)
summary = f"\n{passed}/{len(rows)} checks passed (Python {sys.version.split()[0]}, {sys.platform})"
print(summary)
(HERE / "local_check_report.txt").write_text(summary + "\n\n" + "\n".join(lines), encoding="utf-8")
print("Full output saved to", HERE / "local_check_report.txt")
sys.exit(0 if passed == len(rows) else 1)
