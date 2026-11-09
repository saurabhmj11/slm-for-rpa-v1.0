# OOA autoresearch (agent mode)

Adapted from the idea in [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT): give an AI agent a small
real training setup, a fixed budget and one honest metric, and let it run experiments while you sleep. Here the agent tunes
how our OOA assistant is **trained** (data mix and learning settings), and the judge is our verifier on a frozen flow test.
No agent? `loop.py` does the same search with a scripted strategy (see README in `docs/EXPLAINER.md`).

## Setup (once, with the human)
1. Agree a run tag from today's date (e.g. `nov9`). Create the branch `autoresearch/<tag>`.
2. Read: `recipe.py` (the only file you edit), `constants.py`, `run_experiment.py`, `ooa_eval_frozen.py`, `../ooa_reward.py`.
3. Check the environment: a GPU, nanochat at `NANOCHAT_DIR` with the pretrained base in `NANOCHAT_BASE_DIR`, the platform repo.
4. Create `results.tsv` with the header: `n  score  real_score  solved_rate  val_bpb  minutes  status  description  recipe_json`.
5. Confirm with the human, then start.

## What you CAN do
- Edit `recipe.py` only: data mix (`synth_train`, `flow_repeat`, `smoltalk_rows`, `include_catalog`, `step_weights`,
  `wrap_weights`) and optimisation (`seq_len`, batch, `init_lr_frac`, warm-up / warm-down, `final_lr_frac`).

## What you CANNOT do
- Edit anything else: `constants.py` (fixed token budget), `run_experiment.py`, `ooa_eval_frozen.py`, `frozen/` (hash-checked),
  `../ooa_reward.py`, the catalog, `../gen_flows.py`, `../build_sft_data.py`. They are the referee.
- Train on the frozen eval (the runner removes overlapping rows and reports `leaked_rows_dropped`; do not try to get around it).
- Install packages, or change `token_budget`. Every experiment trains on exactly the same number of tokens.

## The goal
Raise **`score`** (mean verifier reward on frozen generated requests, 0..1). Also watch `real_score` (4 real, long flows: the
reality check) and `solved_rate`. A gain in `score` that makes `real_score` or `valid_rate` fall is suspicious: say so in the
description. Smaller KEEP_EPSILON-level gains are noise; the loop only keeps changes that beat the best by more than 0.003.
**Simplicity criterion:** equal score with fewer rows or fewer changed knobs is a win. Do not keep ugly complexity for a tiny gain.

## The loop
1. Look at `results.tsv` and the current best recipe. Think about WHY the last results went the way they did.
2. Change ONE thing in `recipe.py` (one knob, or a small idea), `git commit`.
3. Run: `python run_experiment.py --platform P --catalog C --workdir W --nanochat N --base-dir B > run.log 2>&1`
4. Read the summary: `grep "^score:\|^real_score:\|^solved_rate:\|^val_bpb:" run.log`. Empty = crash: `tail -n 40 run.log`.
5. Append a row to `results.tsv` (tab-separated; do not commit it): status `keep`, `discard` or `crash`.
6. Better than the best by > 0.003 -> keep the commit (advance the branch). Otherwise `git reset` back.
Crashes (OOM etc.): fix if trivial, else log `crash` and move on. A run over 25 minutes: kill it and count it as a crash.

## Stopping
Unlike an overnight H100, our GPU time is limited (free Colab/Kaggle sessions). Stop when: the human's experiment cap is
reached, three consecutive experiments are discarded after you have tried your best ideas, or the session ends. Before you stop,
write a short summary: best recipe, score gain over baseline, what did NOT help, and your next three ideas.

## Honesty rules
- Report real_score next to score every time. Never claim a win from `score` alone.
- Never edit the evaluator, the frozen set, or the budget to "improve" a number. That voids the run.
- The frozen set has only 40 generated + 4 real requests: small gains can be luck. Prefer changes with a mechanism you can explain.
