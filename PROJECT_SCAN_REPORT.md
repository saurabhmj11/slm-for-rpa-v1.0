# Project Scan Report — "model slm 100 k parameter"

Date: 2026-11-03

## 1. Short answer

**There is no foundation model in this folder.** What exists is:

1. Several **tiny from-scratch models** (about 36K to 1.3M parameters) trained on **synthetic template data**. They work as proofs of pipeline, not as general models.
2. **Reference code from other projects** (nanoGPT, nanochat, llm.c, minbpe). It contains no trained weights, and you did not write it.
3. A **LoRA fine-tune plan** for Qwen2.5-7B (`inhouse-model/`). It is scripts and docs only. It has not been run: there is no `data/` folder and no `.gguf` file.

A foundation model would need billions of tokens of real text, a model of hundreds of millions of parameters or more, and GPU time. None of that has been done here. The practical routes are to:
- fine-tune an existing open model such as Qwen2.5 (already planned in `inhouse-model/`), or
- use nanochat or llm.c to pretrain a small GPT-2-class model from scratch, which needs a GPU budget.

## 2. Inventory

| Area | What it is | Trained weights? | Status |
|---|---|---|---|
| Root: `model.py`, `train.py`, `tokenizer.py`, `data_gen.py`, `infer.py`, `evaluate.py`, `self_improve.py` | "Praman Brain Mini" Layer 1, a News Intelligence agent (sentiment, impact, signal). LLaMA-style decoder with RMSNorm, RoPE and SwiGLU, and one layer block shared across 4 loops. | `best_model_v2.pt` (144 KB), `dpo_model_v2.pt` (144 KB), `ckpt_v2.pt` | Works as a toy |
| `self_improve.py` | A simplified DPO loop that prefers valid JSON tool calls over corrupted ones. | Produces `dpo_model_v2.pt` | Works as a toy |
| `slm logistics/` | Logistics agent: routes a query, makes a tool call, gives an answer. It has a minbpe BPE tokenizer, 15K train and 500 val multi-turn conversations, and a 4.6 MB text dataset. | `logistics_best.pt` (~400 KB) | Works as a toy |
| `slm logistics/model_ctx.py` and `train_ctx.py` | Recurrent-memory transformer in the RMT style: 6 layers, 128-dim, 16 memory slots, 512-token segments. It is built to follow long conversations. | `ctx_best.pt` (5.4 MB, roughly 1.3M parameters) | Trained, **no recorded benchmark result** |
| `inhouse-model/` | QLoRA plan for Qwen2.5-7B on Indian financial data. | None | **Not started** (scripts only) |
| `nanoGPT`, `nanochat`, `llm.c`, `minbpe` | Open-source references. | None | Reference only |
| `gpt_dev.ipynb` | Karpathy-style GPT notebook. | n/a | Learning material |

Parameter counts are **estimated from file sizes**. I could not load the checkpoints, because the Windows Application Control policy blocks `torch\lib\shm.dll` for this Python 3.14 install.

## 3. Findings

### What is good
- The architecture is modern: RMSNorm, RoPE, SwiGLU, weight tying, scaled residual init.
- The recurrent-memory model is a real design idea. It keeps cost per token constant on long conversations.
- Training masks the loss to assistant tokens only, which is correct for agent data.
- There is a memory benchmark: `bench_memory.jsonl` has 630 probes at distances 1 to 32 turns.
- The root `evaluate.py` uses a different seed (123) from training (42).
- There is a DPO stage, and `evaluate.py` has pass thresholds.

### Problems and risks
1. **Stale README.** It describes the first prototype: a ~460-token vocab, a 99,920-parameter model, `best_model.pt`, and a val loss of 0.14. The code and checkpoints are now v2/DPO and Layer 1 (finance news).
   - `best_model.pt` no longer exists, but `debug_infer.py` still loads it, so it will crash.
   - The logistics prototype has no README at all.
2. **Parameter claims do not match.** `model.py` defaults to `vocab_size=100284` (tiktoken), `n_embd=32` and a comment saying "~3M params". The trained v2 files are about 36K parameters, and one shared block is reused 4 times. The "100K parameter" label is not accurate for any current checkpoint.
3. **Self-improvement run never reached its target.**
   - The target val loss was 0.12.
   - It plateaued at about 0.23 in cycle 1, and cycles 2–4 were worse (0.25 to 0.31).
   - Each restart regenerated the data and lowered the learning rate. That changes the data under the comparison, so the cycles cannot be compared fairly.
   - The log stops mid-cycle 5.
   - A loss floor of about 0.23 on template data suggests an irreducible or unlearnable part. Look at which tokens are being missed.
4. **Evaluation is too weak to support accuracy claims.**
   - `evaluate.py` scores only the route and the first observation token. Its own docstring still lists five dimensions.
   - Test cases come from the same template generator as training, so overlap is likely.
   - This measures template memorization, not generalization.
   - No evaluation results are saved anywhere.
5. **Data is purely synthetic and templated.** Entities, phrasing and tool formats are fixed. The model will not handle paraphrases, typos or unseen entities. There is no real user data.
6. **Three separate directions.**
   - Finance news (root).
   - Logistics (`slm logistics/`).
   - Indian finance on Qwen (`inhouse-model/`).
   - They share code but have no common plan or evaluation.
7. **Environment is blocked.** Python 3.14 plus torch hits an Application Control block on Windows, so I could not run training or evaluation. A venv on Python 3.11/3.12, or WSL, is likely needed.
8. **Hygiene.**
   - The folder is not under git, and the ~87 MB `conv_train.jsonl` and the checkpoints are unversioned.
   - There are no tests for the project's own code (the nanochat and minbpe tests are third-party).
   - `self_improve.py` has a mis-encoded character in its header.
   - The `inhouse-model` README promises accuracy gains (for example "~95%") that no run has produced. They are projections, not results.

## 4. Recommendation

Treat this as a **pipeline prototype and decide between two paths**:

- **Path A, the real model (recommended): fine-tune an open base model.**
  1. Pick a base such as Qwen2.5-1.5B or 7B.
  2. Build real data (`inhouse-model/` step 1 already exists).
  3. Run the QLoRA fine-tune on a rented GPU, which the README estimates at about $20 to $30.
  4. Evaluate against the base model.
  This is the only path here that gives a usable language model quickly.
- **Path B, a true from-scratch foundation model.** Use nanochat or llm.c to pretrain a roughly 100M to 1B model. This needs a large text corpus, a GPU budget, and weeks of iteration.
- **Keep the tiny SLMs** as fast routers or tool-callers, not as reasoning models.

### Immediate cleanup (about 1 day)
1. Fix the environment (Python 3.11/3.12 venv) so evaluations can run.
2. Run `evaluate.py`, `evaluate_logistics.py` and `context_bench.py --model ctx` and `--model baseline`. Save the results to files, so the memory model's benefit is measured.
3. Update the README to the real parameter counts, files and results, and delete or fix `debug_infer.py`.
4. Build a held-out test set with paraphrased queries and unseen entities.
5. Put the project under git, with the data and checkpoints ignored or tracked via LFS.
