# RPA Model

In-house conversational AI for the OneOfficeAutomation (OOA) platform. It writes and edits OOA Flow JSON,
operates bots, explains failures and keeps context across a conversation. Built as a learning project on top of
nanochat, with a Qwen2.5 baseline as the yardstick.

Full plan: [docs/PLAN.md](docs/PLAN.md)

## Layout

| Folder | Purpose |
|---|---|
| `docs/` | Plan and design documents |
| `data/raw/` | Source material: platform docs, example flows, conformance cases (copied or linked from the platform repo) |
| `data/catalog/` | Action catalog built from `ooa manifest` and the toolkit docs |
| `data/dialogues/` | Validated multi-turn training dialogues |
| `data/eval/` | Frozen evaluation set. Never train on it |
| `tokenizer/` | OOA-specific BPE tokenizer work (minbpe / nanochat `tok_train`) |
| `training/` | Pretraining, SFT and RL configs and scripts (nanochat based) |
| `eval/` | Scoring: schema validity, action match, memory probes |
| `serving/` | OpenAI-compatible endpoint, retrieval, validate-and-repair loop |
| `notes/` | Learning log: what you ran, what surprised you, what broke |

## Reference code (one folder up)

- `../nanochat-master/` training pipeline we build on
- `../nanoGPT-master/`, `../minbpe-master/`, `../llm.c-master/` learning references
- `../gpt_dev.ipynb` GPT-from-scratch notebook
- Platform repo: https://github.com/OneOfficeautomation-Private-Limited/OneOfficeAutomation-Platform

## Status

Phase 0 in progress. Python 3.12.10 venv in `.venv/` with torch 2.15.0 (CPU only, no CUDA). Next: get nanoGPT
training on Shakespeare (Stage 1 of the learning path).

Activate (PowerShell): `.venv\Scripts\Activate.ps1`
Run without activating: `.venv\Scripts\python.exe script.py`
