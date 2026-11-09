# Track B: in-house model trained with nanochat

Decision (2026-11-03): train the OOA assistant with nanochat (tokenizer, pretrain, mid-train, SFT, optional RL).
Goal is ownership and learning. Expect a small model: it will learn the Flow JSON format, but be weaker than a
fine-tuned Qwen on open-ended requests and long contexts. Track A (Qwen QLoRA, see PLAN.md) is the baseline to compare against.

## Shared pieces (build first, used by any model)
| Piece | Status |
|---|---|
| Action catalog (`data/catalog/catalog.jsonl`, built by `build_catalog.py`) | Done: 1,065 actions, 47 conditions, 12 loop types; counts match the platform's toolkit README |
| Gold flows (48 in `data/raw/`) | Copied (Stage 2) |
| Frozen eval set (`data/eval/`) | Not started. Split by workflow family: `W1-x` and `W1-x-capture` are one family |
| Scorer: `ooa validate` | Needs the platform reference installed (`cd reference && pip install -e ".[test]"`) |

## Track B steps
1. **Tokenizer:** OOA-specific BPE, vocab about 4096 (Stage 2). Minified JSON only. Reweight toward flows.
   nanochat's `tok_train` can replace the minbpe script.
2. **Pretraining corpus:** nanochat expects a large general corpus (FineWeb-style). Your OOA text (about 2 MB) is
   far too small to pretrain on. Decide: general corpus plus OOA text mixed in, or skip pretraining.
3. **Mid-training and SFT data:** validated multi-turn dialogues (create, edit, explain-and-fix, operate, memory
   probes), generated from the catalog and gold flows, kept only if `ooa validate` passes. Tasks plug in through
   nanochat's task/conversation format.
4. **Context length:** median flow is about 2,000 tokens, so train with a window of 4096 or more. nanochat's
   sliding-window layers (`SSSL`) make this cheaper.
5. **Compute:** the CPU-only machine only runs the tiny demo. Real runs need a rented GPU node.
6. **Eval:** schema validity, action match, memory probes, against the frozen set, compared with Track A.

## Open blockers
- nanochat's task/SFT data format still to be read (cloned in the cloud session, not yet pinned to a commit).
- Data policy: may docs and flows go to a cloud LLM to generate dialogues?

## Update: environment checks (cloud session)
- nanochat cloned and platform reference installed. `ooa manifest` works.
- `ooa validate` (Python reference) only knows a few toolkits, so it fails 42 of 47 valid use-case flows
  ("Package ooa.browser is not installed"). It cannot be the main scorer.
- Scorer to use instead: `eval/flow_check.py` = JSON schema (`flow.v1.json`) + every action and input name exists
  in `catalog.jsonl`. All 48 gold flows pass it with 0 unknown actions or inputs. Add variable-declaration and
  reference checks next; keep `ooa validate` as an extra check for the toolkits it supports.
- Two gold flows disagree with the reference on excel inputs (`header` on `excel/open`, `save` on `excel/close`):
  the docs and the Python reference are not fully in sync. Treat the toolkit docs (catalog) as the contract.

## nanochat SFT format (read from commit 92d63d4)
- A `Task` yields `{"messages": [...]}`: optional system message, then alternating user/assistant string turns.
  `render_conversation` masks loss to assistant tokens. Special tokens: user/assistant start/end, python and
  output start/end (used for tool calls, as GSM8K does).
- `scripts/chat_sft.py` hardcodes its mixture: SmolTalk (460K) + MMLU x3 + GSM8K x4. For OOA, add
  `training/ooa_task.py` (`OOADialogues`, reads JSONL) to that list, and reduce or drop the general tasks.
- Default `--max-seq-len` is inherited from pretraining (2048 in the speedrun config). Median flow is about 2,000
  tokens at vocab 4096, so pretrain with a longer window, or SFT will truncate large flows.
- nanochat's own tokenizer is also BPE (rustbpe/tiktoken), with 9 special tokens; the Stage 2 vocab-4096 experiment
  can be redone with `scripts/tok_train.py`.

## Colab notebook (end to end)
`training/ooa_nanochat_colab.ipynb` (regenerate with `training/make_colab_notebook.py`): setup, tokenizer, pretrain on a
free T4 (resumes from Drive after disconnects), seed SFT data from the platform repo (`build_sft_data.py`),
patched `chat_sft.py`, held-out test with schema + catalog scoring. Profiles: `smoke` (pipeline check), `t4`, `a100`.
Written and syntax-checked without a GPU: not yet run on Colab. Seed data: about 60 flow dialogues (x3 wordings) plus
about 2,100 catalog dialogues; 5 held-out eval prompts, so results are only a smoke signal.

## Reinforcement layer (self-learning and self-improving)
All loops are scored by a **verifier**, not a human or an LLM judge: `training/ooa_reward.py`
(schema 0.15, actions exist 0.15, required inputs and names 0.10, declared variables and unique ids 0.10, valid JSON 0.20,
object start 0.05, action sequence vs the gold flow 0.25 with partial credit). Empty flows score <= 0.25, so they cannot
win. `training/test_reward.py` checks this (run: `python test_reward.py PLATFORM_REPO`).
- **RL (notebook cell 9):** nanochat's own `chat_rl.py`, patched to use `ooa_rl_task.py`. 8 answers per request;
  answers above the group average are reinforced (REINFORCE, no KL; as nanochat implements it).
- **Self-improvement rounds (cell 10, `ooa_self_improve.py`):** fresh requests each round, K attempts each; verified wins
  go to `self_improved.jsonl`, failures go to `hard_fixed.jsonl` (with the verified gold flow) and are retried. Then SFT
  retrains from the pretrained base on everything. Progress = pass@k on fresh requests, logged in `selfimprove_log.jsonl`.
- **Real usage (`ingest_feedback.py`):** accepted or human-edited flows from Studio become training data only if they
  also pass the verifier; rejected-without-edit prompts are listed as hard cases.
Status: reward, generator, ingest and the notebook patches are tested here against the real nanochat files; the RL and
sampling scripts have NOT been run on a GPU. Risks: the reward only knows what the checker knows (it cannot tell if a
flow does what a user meant), RL on a weak model gets little signal (hence partial credit and a short-first curriculum),
and the self-improvement loop only grows as far as the generator's request styles reach.

## First t4 results (2026-11-07) and what they showed
- Pretraining finished (4577 steps). SFT was only 100 steps (1.8 min), about 3M tokens: the cap was SFT_ITERS=400, which nanochat counts in micro-batches (4 per optimizer step), so 400 -> 100 steps (an earlier note here wrongly blamed the data running out). Fix: SFT now
  uses 30,000 generated flows and up to 3,000 steps for non-smoke profiles.
- The old validation set overlapped training, so its bpb (0.07) was meaningless. The validation set now holds only
  unseen catalog rows and unseen generated flows.
- Self-improvement rounds: fresh-request pass@8 went 5% -> 10% on 20 requests per round. That is within noise; the loop
  added about 22 rows to 13,400. A 20-request round took about 3 minutes (so about 15 min for 100 requests).
- The round log now also reports the average of each reward part, to show why answers fail (invalid JSON, wrong actions,
  wrong order, missing inputs).

## Round table with per-part scores (2026-11-07)
Average reward part over all samples (1.0 = always fine): valid JSON 0.73-0.75, schema 0.53-0.56, actions exist 0.22,
required inputs correct 0.03, variables declared 0.50, action sequence matches 0.25-0.29. Fresh-request pass@8: 5-10%.
Reading: the model has learned the shape of a flow but invents action and input names. RL and self-improvement cannot
fix that efficiently. Next step: retrieval, so the prompt carries the candidate actions (ids and input names) to copy.
Cell 10 now stops if the generated data or patched SFT script do not match the config (it silently reused old ones).

## Performance fix (2026-11-07)
The data cell took 35+ min on Colab because jsonschema.validate() rebuilds its validator on every call (about 70 ms per
flow). The validator is now built once: 30,000 generated flows, each fully schema-checked, take about 1 minute; the reward
scorer is about 45x faster too (this also speeds up RL and self-improvement sampling).

## Kaggle version
`training/make_kaggle_notebook.py` turns the Colab notebook into `ooa_nanochat_kaggle.ipynb` (no Drive; files in
/kaggle/working; internet check; restores `nanochat_base` from /kaggle/input; last cell packages the model as a zip).
Syntax-checked and the packaging cell tested on fake files; NOT yet run on Kaggle. Kaggle notes: GPU T4 x2 or P100
(this notebook uses one GPU), about 30 GPU hours per week, session up to 12 h, Internet must be On (phone verification).

## Retrieval result and the end-to-end system (2026-11-07)
`training/retrieval.py` (BM25 over the catalog) + `training/eval_retrieval.py`. On human-written flow descriptions, top-10
finds only ~10% of the actions real flows use (0/31 flows fully covered); offering all actions of the top-3 toolkits covers
25% (114 candidates). Real flows average 18.5 steps, 11 distinct actions, 3.7 toolkits; the generated flows (1-4 steps, one
toolkit) are far easier, so synthetic scores do NOT predict real performance. Consequence: our small model is for simple
requests; complex ones go to Claude, with the whole best-matching toolkits in the prompt.
Built and tested: `serving/` (OpenAI-compatible server, router, repair, verifier, PII masking, client policy switches,
feedback endpoint; 8 end-to-end tests with fake models), `training/repair.py`, `ooa_reward.problems()`.

## Autoresearch (2026-11-09)
`training/autoresearch/`: adapted from karpathy/autoresearch (MIT): single editable file (`recipe.py`), fixed budget (nominally 12M tokens, really 3.0M tokens = 92 optimizer steps, see correction below)
(`constants.py`), frozen scoreboard (`frozen/`, sha256-checked: 40 generated + 4 real flows, with its own catalog snapshot),
read-only runner (`run_experiment.py`: builds data, drops rows overlapping the frozen set, trains, scores), scripted
hill-climbing (`loop.py`, results.tsv, resumable) and agent instructions (`program.md`). Tested here: loop logic with a fake runner,
runner dry-run, guards (budget and unknown-key edits refused), embedded scoreboard integrity. NOT run on a GPU yet.
Notebook cell 12 runs it. Metric: score = mean verifier reward on frozen generated requests; real_score = on 4 real flows.

## Result after the bigger SFT (2026-11-09, t4, step 750)
Unseen generated requests (20): valid JSON 20/20, schema 20/20, actions exist 20/20, exact action sequence 10/20 (before: 4, 3, 1, 0).
Held-out real flows (5): 0 usable (memorised-flow recitation; one prompt produced chat text). Catalog lookup answered correctly.
Added notebook cell 13 (whole assistant, local only: model -> repair -> verify), logic tested with a stub model; the real local backend
has not run yet. Note: real gold flows are repeated 15x in training (flow_repeat), which likely encourages memorisation; the autoresearch
loop can test lower values.


## Correction: the autoresearch budget is 3M tokens, not 12M (2026-11-10)
`chat_sft.py` stops after `--num-iterations` MICRO-batches (`it` counts one device batch each), and an optimizer step is grad_accum=4
micro-batches. We passed 366 meaning optimizer steps, so every experiment trained 366/4 = 92 steps = 3.0M tokens (a user-uploaded
`meta_000092.json` showed "step": 92). The runner's old `train_tokens_M: 12.0` was computed, not measured. Comparisons between experiments
stay valid (all got the same 3.0M), but conclusions apply to a SHORT training regime. Also, experiment #1 (8,000 -> 30,000 generated flows)
did not show "variety beats repetition": 3M tokens is only a slice of the shuffled mixture either way, so it changed the PROPORTIONS in
that slice (generated flows about 60% -> about 81% of rows; catalog Q&A about 19% -> 7%), which also explains val_bpb rising (catalog loss).
The same micro-batch rule applies to the main SFT cell: SFT_ITERS=3000 gives 750 steps (about 24M tokens); raise SFT_ITERS to train longer.
