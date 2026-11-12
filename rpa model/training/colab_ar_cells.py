"""Notebook cell for the autoresearch loop (imported by make_colab_notebook.py)."""


def build(md, code, frozen_blob):
    md("""
## Autoresearch: the training recipe improves itself

Idea from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT): fixed budget, one honest metric, change one
thing, keep it only if it helps. Here each experiment re-trains the SFT stage for a fixed **3M tokens (92 optimizer steps)** with one change to
the recipe (data mix or learning settings), then scores it on a **frozen** test of 40 generated + 4 real flows using our
verifier. Better than the best by more than 0.003 -> kept; otherwise discarded. Resumable: results are saved to Drive/working dir.

Each experiment takes about 10 minutes on a T4 (my estimate: 6-8 minutes training + 3 minutes scoring; not yet measured).
Run it after SFT works (cells 6-8). With an AI agent instead of this scripted search, see `autoresearch/program.md`.
""")
    code('''
#@title 12. Autoresearch: scripted search over the SFT recipe (fixed 3M-token budget, frozen scoreboard)
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

import shutil, time
live = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatsft_checkpoints")
_keep = None                                    # your real model is moved aside now and put back at the end, even if stopped
if os.path.exists(live) and not os.path.exists(f"{live}/.autoresearch"):
    _keep = live + "_kept_during_autoresearch_" + time.strftime("%%Y%%m%%d_%%H%%M%%S")
    shutil.move(live, _keep)
    print("your model is kept safe in", _keep, "and is put back when this cell ends")
def save_best(recipe, score):                       # the model just trained is the best so far: keep a copy on Drive
    shutil.rmtree(f"{work}/best_model", ignore_errors=True)
    shutil.copytree(live, f"{work}/best_model")
    json.dump({"score": score, "recipe": recipe}, open(f"{work}/best_model/best_recipe.json", "w"))
try:
    _rows = loop.read_results(f"{work}/results.tsv")
    if _rows and not os.path.exists(f"{work}/best_model"):
        # an earlier session found a best recipe but its model was not saved: train it once more (also measures run-to-run noise)
        _top = [r for r in _rows if r["status"] in ("baseline", "keep")][-1]
        print(f"Rebuilding the best model: re-training the best recipe once (logged score {_top['score']})")
        _res = runner(json.loads(_top["recipe_json"]))
        loop.append_result(f"{work}/results.tsv", {"n": len(_rows), "score": f"{_res['score']:.4f}", "real_score": _res.get("real_score", ""),
            "solved_rate": _res.get("solved_rate", ""), "val_bpb": _res.get("val_bpb", ""), "minutes": _res.get("minutes", ""),
            "status": "recheck", "description": f"re-train of best (logged {_top['score']}): run-to-run noise = {_res['score'] - float(_top['score']):+.4f}",
            "recipe_json": _top["recipe_json"]})
        save_best(json.loads(_top["recipe_json"]), _res["score"])
        print(f"re-trained score {_res['score']:.4f} vs logged {_top['score']}  (difference = run-to-run noise)")
    best, best_score = loop.run_loop(runner, RECIPE, f"{work}/results.tsv", max_experiments=AR_EXPERIMENTS, on_keep=save_best)
finally:
    # the live model goes back to YOUR model; experiments are short (92 steps) and only say which recipe to use next
    if _keep and os.path.exists(_keep):
        shutil.rmtree(live, ignore_errors=True)
        shutil.move(_keep, live)
        print("your model is back in", live)
    print("best experiment model (for comparison only):", f"{work}/best_model")
print("\\nBest recipe so far:", json.dumps(best))
print("\\nAll experiments:")
for r in loop.read_results(f"{work}/results.tsv"):
    print(f"#{r['n']:>2} {r['status']:8s} score {r['score']:>7} real {r['real_score']:>7} | {r['description']}")
''' % frozen_blob)
