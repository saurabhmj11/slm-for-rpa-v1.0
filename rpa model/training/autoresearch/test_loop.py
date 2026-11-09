"""Tests the search logic with a fake runner (no GPU): python test_loop.py"""
import json, os, tempfile
import loop
from recipe import RECIPE

calls = []
def fake(recipe):
    calls.append(loop.rhash(recipe))
    if recipe["seq_len"] == 1024:
        raise RuntimeError("out of memory")                 # a crashing config
    s = 0.40
    s += 0.05 * (recipe["init_lr_frac"] == 0.5)            # hidden truth: lower LR helps
    s += 0.03 * (recipe["synth_train"] == 16000)           # more data helps
    s -= 0.02 * (recipe["flow_repeat"] == 30)              # too much repetition hurts
    s += 0.001 * (recipe["warmdown_ratio"] == 0.8)         # below the noise threshold: must NOT be kept
    return {"score": s, "real_score": 0.02, "solved_rate": s / 4, "val_bpb": 0.1, "minutes": 9.0}

d = tempfile.mkdtemp(); path = d + "/results.tsv"
saved = []
best, score = loop.run_loop(fake, RECIPE, path, max_experiments=60, seed=3, on_event=lambda m: None,
                          on_keep=lambda r, sc: saved.append(sc))
rows = loop.read_results(path)
status = [r["status"] for r in rows]
assert rows[0]["status"] == "baseline" and status.count("baseline") == 1
assert best["init_lr_frac"] == 0.5 and best["synth_train"] == 16000, best
assert best["flow_repeat"] != 30 and best["warmdown_ratio"] != 0.8, "noise-level or harmful changes must be discarded"
assert "crash" in status and "keep" in status and "discard" in status
kept = [float(r["score"]) for r in rows if r["status"] in ("baseline", "keep")]
assert kept == sorted(kept), "the best score never goes down"
assert len(set(calls)) == len(calls), "no experiment is ever repeated"
assert len(saved) == status.count("baseline") + status.count("keep") and saved == sorted(saved), "on_keep fires for the baseline and every keep"
print(f"search ok: {len(rows)} experiments, best score {score:.3f}, kept {status.count('keep')}, crashes {status.count('crash')}")

# resume (separate run, stopped part-way): finished experiments are not repeated, new ones continue where it left off
calls.clear(); path2 = d + "/results2.tsv"
loop.run_loop(fake, RECIPE, path2, max_experiments=10, seed=3, on_event=lambda m: None)
assert len(calls) == 10
loop.run_loop(fake, RECIPE, path2, max_experiments=10, seed=3, on_event=lambda m: None)
assert len(calls) == 10, "resume must not rerun finished experiments"
loop.run_loop(fake, RECIPE, path2, max_experiments=15, seed=3, on_event=lambda m: None)
assert len(calls) == 15 and len(set(calls)) == 15 and len(loop.read_results(path2)) == 15
print("resume ok: 10 done -> rerun does nothing -> 5 more run, none repeated")
print("ALL LOOP TESTS PASSED")
