"""Scripted autoresearch loop (no AI agent needed): greedy hill-climb over the SFT recipe, one change per experiment.

 baseline -> propose ONE mutation of the current best recipe -> run a fixed-budget experiment -> score on the frozen eval ->
 keep it only if it beats the best by more than KEEP_EPSILON, else discard -> log -> repeat.
Resumes from results.tsv (re-running skips experiments already done). `runner(recipe) -> dict` is injected, so the logic
is tested without a GPU. results.tsv columns: n, score, real_score, solved_rate, val_bpb, minutes, status, description, recipe_json
"""
import copy, csv, hashlib, json, os, random, time
from constants import KEEP_EPSILON

SEARCH_SPACE = {
    "init_lr_frac":   [0.3, 0.5, 0.8, 1.2, 1.6],
    "warmdown_ratio": [0.2, 0.5, 0.8],
    "warmup_ratio":   [0.0, 0.05],
    "final_lr_frac":  [0.0, 0.1],
    "synth_train":    [4000, 8000, 16000, 30000],
    "flow_repeat":    [5, 15, 30],
    "smoltalk_rows":  [0, 1500, 3000],
    "include_catalog": [True, False],
    "step_weights":   [[25, 35, 25, 15], [10, 30, 30, 30], [40, 40, 15, 5], [0, 25, 35, 40]],
    "wrap_weights":   [[55, 17, 14, 14], [30, 25, 20, 25], [80, 7, 7, 6]],
}
# Informed order: tried first (each costs ~9 minutes). Reasons come from the first results: a LOWER learning rate hurt, so try
# HIGHER; real flows may be memorised through flow_repeat=15; dropping general chat rows leaves more budget for flows.
# Deliberately NOT here: include_catalog=False (would likely raise the score by deleting catalog Q&A, which the frozen test does not measure).
PRIORITY = [("init_lr_frac", 1.2), ("init_lr_frac", 1.6), ("flow_repeat", 5), ("smoltalk_rows", 0),
            ("warmdown_ratio", 0.8), ("warmup_ratio", 0.05), ("flow_repeat", 30)]
COLS = ["n", "score", "real_score", "solved_rate", "val_bpb", "minutes", "status", "description", "recipe_json"]


def rhash(recipe):
    return hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]


def read_results(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_result(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


def mutations(best, tried, rng):
    out = []
    for knob, values in SEARCH_SPACE.items():
        for v in values:
            if best.get(knob) == v:
                continue
            cand = copy.deepcopy(best)
            cand[knob] = v
            if rhash(cand) not in tried:
                out.append((knob, v, cand))
    rng.shuffle(out)
    rank = {kv: i for i, kv in enumerate(PRIORITY)}
    out.sort(key=lambda t: rank.get((t[0], t[1] if not isinstance(t[1], list) else tuple(t[1])), len(PRIORITY)))   # stable: priority first, rest stay shuffled
    return out


def run_loop(runner, base_recipe, results_path, max_experiments=20, seed=0, on_event=print, on_keep=None):
    rng = random.Random(seed)
    rows = read_results(results_path)
    tried, best, best_score = set(), copy.deepcopy(base_recipe), None
    for r in rows:                                   # replay history so a resumed run continues where it stopped
        rec = json.loads(r["recipe_json"])
        tried.add(rhash(rec))
        if r["status"] in ("keep", "baseline") and r["score"]:
            best, best_score = rec, float(r["score"])
    done = len(rows)
    if best_score is None:                           # first run: the unchanged recipe is the baseline
        res = runner(best)
        append_result(results_path, {"n": done, "score": res["score"], "real_score": res.get("real_score", ""),
                                     "solved_rate": res.get("solved_rate", ""), "val_bpb": res.get("val_bpb", ""),
                                     "minutes": res.get("minutes", ""), "status": "baseline", "description": "baseline",
                                     "recipe_json": json.dumps(best, sort_keys=True)})
        best_score, done = res["score"], done + 1
        tried.add(rhash(best))
        if on_keep:
            on_keep(best, best_score)
        on_event(f"baseline score {best_score:.4f}")
    while done < max_experiments:
        cands = mutations(best, tried, rng)
        if not cands:
            on_event("search space exhausted"); break
        knob, value, cand = cands[0]
        tried.add(rhash(cand))
        desc = f"{knob}: {best.get(knob)} -> {value}"
        t0 = time.time()
        try:
            res = runner(cand)
            status = "keep" if res["score"] > best_score + KEEP_EPSILON else "discard"
        except Exception as ex:                      # OOM, bad config, ...: log and move on
            res, status = {"score": 0.0}, "crash"
            desc += f" ({str(ex)[:80]})"
        append_result(results_path, {"n": done, "score": f"{res['score']:.4f}", "real_score": res.get("real_score", ""),
                                     "solved_rate": res.get("solved_rate", ""), "val_bpb": res.get("val_bpb", ""),
                                     "minutes": res.get("minutes", round((time.time() - t0) / 60, 1)), "status": status,
                                     "description": desc, "recipe_json": json.dumps(cand, sort_keys=True)})
        if status == "keep":
            best, best_score = cand, res["score"]
            if on_keep:
                on_keep(best, best_score)
        on_event(f"#{done} {desc} -> {res['score']:.4f} [{status}] (best {best_score:.4f})")
        done += 1
    return best, best_score
