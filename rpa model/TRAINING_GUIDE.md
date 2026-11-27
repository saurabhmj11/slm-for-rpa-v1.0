# Training guide (Colab + your laptop)

Plain steps to train the OOA model again. Everything stays private: this repository, your Google Drive folder
and the models are yours. The notebooks in the repository have the Drive folder **empty** on purpose: paste your
folder id into cell 1 (`SHARED_FOLDER_ID` on Colab, `DRIVE_FOLDER_ID` on the laptop), or use the personal copies
you were sent, which already have it.

| Where | Notebook |
|---|---|
| Colab | `rpa model/training/ooa_nanochat_colab.ipynb` |
| Laptop (Windows, NVIDIA GPU) | `rpa model/ooa_local_pc.ipynb` |

Every new model is tested on the same 139 requests and **installed only if it beats the current one** (the old
model is always kept). Training stops by itself on `nan` and never saves a broken model.

---

## 1. Upload the new training data to Drive (once)

Upload these three files from `rpa model/data/extra_sft/` (or from `new_sft_data.zip`) to Drive
**`ooa-nanochat/sft/`**, replacing older copies:

- `decomposition_train.jsonl`: long multi-app business processes broken into steps, plus single-step parts
- `wording_train.jsonl`: Indian English, story-style, voice, Hinglish, typos, short requests, questions for missing details
- `session_train.jsonl`: sign in / connect once, later steps use the session

Keep only one copy of each (delete any `name (1).jsonl`).

## 2. Colab: SFT with the new data (about 4-5 h on a T4, saved every ~20 min)

1. Upload/open the Colab notebook. *Runtime → Change runtime type → GPU* (T4 is fine; L4/A100 are faster).
2. Cell 1: leave the defaults (`PROFILE = "t4"`, `SFT_CHUNKS = 12`, `RESTART_SFT = False`).
3. Run **1 → 2 → 6 → 6d → 7** (skip 3-5: the base model is already on Drive).
   - 6d renders the new files once (10-20 min); it must not print `MISSING … .jsonl`.
   - Cell 7 prints `NEW SFT RUN` and the mixture table lists the three files above.
4. **Session ended?** Run **1 → 2 → 6 → 6d → 7** again (also from your second Google account, if it has Editor
   access to the folder and a shortcut to it in My Drive). Cell 7 prints `CONTINUING … Next: chunk N`.
   Only one session at a time.
5. At the end: `SFT FINISHED` and `all weights are finite (checked)`. Run **cell 8** and keep its result lines
   (set 1/2/3, beginner, modes table). Then *Runtime → Disconnect and delete runtime* to save your hours.

What to watch in cell 7: `loss` is a number (roughly 0.01-0.4, it jumps between files: normal),
`Validation bpb` is a small number, never `nan`. An `OOA SKIP` line means one bad step was skipped: fine.
`OOA NAN STOP`: the chunk stopped without saving; it is retried once automatically.

Leave `RL_V5 = False` (cell 9 prints "Skipped": that is expected).

## 3. Laptop: install / test a Colab model (about 20 min)

1. Unzip the PC package (`ooa_rpa_model_pc.zip`) over your "rpa model" folder (or rename the old folder to
   `rpa model OLD` first; cell 2 copies your models and the `platform` folder over).
2. Cell 1: `MODE = "install"`. Run **1 → 2 → 3 → 6**. Cell 6 prints the scores and which model it installed.
3. Cell 7 starts the assistant with the installed model.

## 4. Laptop: clean base model v6 (about 12-14 h, overnight, GPU only)

Licence-clean pretraining (FineWeb-Edu ODC-By + your own platform material; no ClimbMix), new tokenizer.

1. Measure first (about 15 min). Cell 1: `MODE = "pretrain"`, `PRETRAIN_MEASURE = 30`. Run **1 → 5**.
   Check the `MEASURED: … about H h` line (11-19 h is normal; out of memory → send the output).
2. Real run: `PRETRAIN_MEASURE = 0`, run **1 → 5**. Saved every 200 steps; after any stop run 1 → 5 again.
   Loss starts near 9.7, falls below 5 within a few hundred steps, then slowly.
3. Done when it prints `PRETRAINED: …train_data/base_v6 (step 5760 …)`.

Laptop checklist: plugged in, *Best performance* power mode, airflow, Windows Update restarts paused.

## 5. Laptop: SFT on the clean base (about 4 h)

Cell 1: `MODE = "full"`, `BASE = "v6"`. Run **1 → 6**. SmolTalk is off (licence). Cell 6 tests and installs if better.

## 6. Optional after a good model

| Step | Cell 1 | Time on an RTX 4060 |
|---|---|---|
| Refine on the weak spots | `MODE = "refine"` | about 30 min |
| RL (the platform checker scores answers) | `MODE = "rl"` | about 6 h |
| Finish a Colab SFT run on the laptop | `MODE = "colab"` | about 20 min per 1000 steps |

## Troubleshooting

| You see | Do |
|---|---|
| `MISSING in Drive sft/ … .jsonl` (6d) | upload that file to `ooa-nanochat/sft/`, run 6d again |
| `NEW SFT RUN` when you expected `CONTINUING` | stop: the session sees a different Drive folder (check cell 2's folder line) |
| Drive full | empty Drive's trash (deleted chunks still count), run cell 7 again |
| `STOPPED … rerunning will not help` | send the last 30 lines of the cell |
| `NOT TESTED, NOT INSTALLED: … nan` | the model is broken; it was not installed; your old model stays |
| Colab "runtime may last N hours" < 1 h | let the current chunk finish, then disconnect; continue later |

## Licences (keep with the product)

- FineWeb-Edu (clean base): ODC-By 1.0, also subject to the Common Crawl Terms of Use: keep the attribution line
  from `training/pretrain/DATA_CARD.md`.
- The current (old) base was pretrained on ClimbMix (CC BY-NC 4.0, non-commercial): the clean base replaces it.
- SmolTalk rows (old SFT mix) carry Llama 3.1 licence conditions: off for the clean base (`SMOLTALK_ROWS = 0`).
- Some training examples were written with an AI assistant; have counsel confirm its terms for your use.
