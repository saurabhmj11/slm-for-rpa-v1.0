"""Notebook cell for the autoresearch loop (imported by make_colab_notebook.py)."""


def build(md, code, frozen_blob):
    md("""
## Autoresearch: the training recipe improves itself

Idea from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT): fixed budget, one honest metric, change one
thing, keep it only if it helps. Here each experiment re-trains the SFT stage for exactly **12M tokens** with one change to
the recipe (data mix or learning settings), then scores it on a **frozen** test of 40 generated + 4 real flows using our
verifier. Better than the best by more than 0.003 -> kept; otherwise discarded. Resumable: results are saved to Drive/working dir.

Each experiment takes about 10 minutes on a T4 (my estimate: 6-8 minutes training + 3 minutes scoring; not yet measured).
Run it after SFT works (cells 6-8). With an AI agent instead of this scripted search, see `autoresearch/program.md`.
""")
    code('''
#@title 12. Autoresearch: scripted search over the SFT recipe (fixed 12M-token budget, frozen scoreboard)
AR_EXPERIMENTS = 6          # experiments this session (including the baseline); about 10 min each
import base64, json, os, subprocess, sys, zlib
AR = "/content/ooa/autoresearch"
os.makedirs(AR + "/frozen", exist_ok=True)
_blob = "%s"
for _n, _t in json.loads(zlib.decompress(base64.b64decode(_blob))).items():
    open(f"{AR}/frozen/{_n}", "w", encoding="utf-8").write(_t)
sys.path.insert(0, AR)
import loop
from recipe import RECIPE
work = f"{DRIVE_DIR}/autoresearch"
os.makedirs(work, exist_ok=True)

def runner(recipe):
    open(f"{work}/recipe_try.py", "w").write("from constants import FIXED_TOKEN_BUDGET\\nRECIPE = " + repr(recipe) + "\\n")
    cmd = [sys.executable, f"{AR}/run_experiment.py", "--platform", "/content/platform", "--catalog", f"{DRIVE_DIR}/sft/catalog.jsonl",
           "--workdir", f"{work}/run", "--nanochat", "/content/nanochat", "--base-dir", os.environ["NANOCHAT_BASE_DIR"],
           "--recipe", f"{work}/recipe_try.py", "--ooa-dir", "/content/ooa"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout[-600:])
    if p.returncode != 0:
        raise RuntimeError((p.stdout + p.stderr)[-300:])
    return json.load(open(f"{work}/run/result.json"))

best, best_score = loop.run_loop(runner, RECIPE, f"{work}/results.tsv", max_experiments=AR_EXPERIMENTS)
print("\\nBest recipe so far:", json.dumps(best))
print("\\nAll experiments:")
for r in loop.read_results(f"{work}/results.tsv"):
    print(f"#{r['n']:>2} {r['status']:8s} score {r['score']:>7} real {r['real_score']:>7} | {r['description']}")
''' % frozen_blob)
