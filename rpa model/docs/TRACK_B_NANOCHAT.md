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
- nanochat source is not in this repo; cloning it into the cloud session was denied, so its current script and
  task APIs have not been checked here. Pin a commit when it is added.
- Platform reference is not installed here for the same reason, so `ooa validate` has not been run.
- Data policy: may docs and flows go to a cloud LLM to generate dialogues?
