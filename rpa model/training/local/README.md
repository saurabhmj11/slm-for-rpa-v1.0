# Train the OOA model on a PC (Colab v4 recipe)

The easiest way is the notebook **`rpa model/ooa_local_pc.ipynb`** (open it in VS Code or Jupyter, run the cells
from top to bottom). It drives the three scripts in this folder, which reuse the Colab code (one source of truth:
`training/sft_setup.py`, `render_dataset.py`, `data_filter.py`, `ooa_format.py`):

| Script | What it does |
|---|---|
| `data_pc.py` | Downloads from Google Drive (`ooa-nanochat`, shared as "anyone with the link") into `rpa model/train_data/`: the rendered data `sft_render/` (when Drive has one that changed since the last download), else `sft/` (+ `sft_clean/`) and renders it here with the Colab cell 6 / 6d settings (data_filter, then render_dataset with the catalog, `--max-chars 13000 --max-tokens 4068 --answer-format compact --gold-dropout 0.5` and the new files long, business2, targeted, fix, clarify, beginner); the newest trained model (`chatsft_checkpoints/d8`, else the newest `chatsft_checkpoints_before_*`); the pretrained base model for full training. Drive files named `name (1).jsonl` are second uploads: reported and ignored. For a top-up it also renders the assistant's Accept / Edit feedback into `feedback_train.jsonl` in the same format (the full run trains on the Colab data only). |
| `train_pc.py` | Trains with nanochat's `scripts/chat_sft.py` from the pinned commit, patched by `sft_setup.patch_chat_sft` (the 4096 attention-window fix, the mixture planned for the step budget, rows longer than the window skipped, "Training data actually read" at the end), in saved chunks that resume after any stop. |
| `eval_pc.py` | Scores the new model and the installed one through the full assistant (`serving/ooa_server.Router`: catalog lock, repair, verify, 2 self-corrections) on sets 1-3 (`eval/plain_requests*.jsonl`) and the beginner set (`eval/beginner_requests.jsonl`), like Colab cell 8, and installs the new model only if it is better. |
| `topup_nightly.bat` | Windows: the top-up every night at 01:30 (continues the run, then tests and installs). |
| `setup_wsl.sh` | Optional: a Python environment in WSL / Linux, where `torch.compile` works (faster full training on a GPU). |

## Modes (cell 1 `MODE`)

- **`full`** (default with an NVIDIA GPU): exactly Colab cell 7. 8000 steps of 32,768 tokens at the 4096 window from
  the **pretrained base model**, the full mixture planned for that budget (`sft_setup.plan_for_notebook`), 3000
  SmolTalk rows, learning rate 0.8 of the base one. The device batch follows the GPU memory (2 x 4096 on 14+ GB like
  the Colab T4, 1 below, 4 on 30+ GB); the tokens per step stay the same. About 2-3 hours on an RTX 3060/4060.
  Needs an RTX 20xx / GTX 16xx or newer. On plain Windows it runs without `torch.compile` (slower); WSL compiles.
  nanochat saves only at the end of a run, so by default (`FULL_CHUNKS = 1`) a stop means starting again; with
  `FULL_CHUNKS = 8` the run is saved every 1000 steps and resumes, with the same learning-rate schedule (each chunk
  continues the previous chunk's model and optimizer; flat lr over the first half, then the linear warmdown; the Muon
  momentum ramp counts the run's steps, not the chunk's) and the same planned mixture read with a new shuffle per chunk.
- **`topup`** (default without a GPU): continues the **newest trained model** (Drive's newest, with its optimizer
  state when Drive has it; the PC's own when it already grew from that same Drive model in an earlier top-up: the
  model is known by its Drive file id, so it stays the same model when Colab moves it to
  `chatsft_checkpoints_before_*`) on the same rendered data, with the window fix, at the 4096 window (2048 when the PC
  has less than 10 GB RAM; `train_pc.py` and the nightly task apply the same rule when `--seq` is not given), learning rate 0.3, for `TOPUP_STEPS` (default 300)
  steps, saved every 50 steps, at low priority with one core left free. A top-up is about 1/25 of the Colab budget,
  so its mixture is not the full one spread thin (every file at 0.1-0.3 views would teach little):
  `sft_setup.plan_topup` gives 35% of the tokens to the beginner rows, 20% each to the long and business2 processes,
  6% clarify, 4% targeted, 3% fix (2% feedback when there is some) and replays every other file with the last 10% in
  the full mix's proportions so nothing learned is forgotten.
  Measured on a 4-core test machine (no GPU, no `torch.compile`, as on Windows): **55 s per step at the 4096 window**
  with all 4 threads and **67 s at low priority with 3 threads (the default)**, so 300 steps take about 5.6 hours
  there (plus about 3 minutes of loading per 50-step chunk); a slower 4-core PC (the old 2048 runs took 77 s per
  step on the user's PC) needs up to about 9 hours. `TOPUP_HOURS` in cell 1 measures this
  PC first (about 5 minutes, at the same low priority) and sizes every new round to that many hours, chunk loading
  included.
- Drive's `sft_render/` is used only when its `render_report.json` has the Colab cell 6d settings for this PC's window
  (`answer_format` compact, `gold_dropout` 0.5, `max_tokens` window - 28) and every new file (long, business2,
  targeted, fix, clarify, beginner); otherwise (an older render, or a 2048 PC) the data is rendered here.
- Stopping cell 5 or 6 ends the step with everything it started (on Windows the whole process tree; `train_pc.py`
  also puts the training process in a Windows job that ends with it).
- **`install`**: no training; tests Drive's newest model against the installed one and installs it if it is better
  (the quickest way to get a new Colab model onto the PC).

The old modes `long`, `refine`, `curriculum` and `big` (and their scripts `train_long.py`, `refine_local.py`,
`curriculum_local.py`, `big_local.py`, `train_local.py`, `topup_local.py`) are gone: they trained the old plain-prompt
format with plain-JSON answers and a 2048 window without the window fix, so their result would undo the v4 model.

## Run order

1. Settings (`DRIVE_FOLDER_ID`, `MODE`). 2. Check this PC (installs PyTorch and the packages). 3. Download (and
render when Drive has no `sft_render`, 30-60 minutes). 4. Check (shows the plan and the data mixture).
5. Train (resumable: after a stop run cells 1-5 again). 6. Test and install. 7. Start the assistant.

## Testing and installing (cell 6)

The installed model is scored with what the assistant gives it now (`rpa model/sft/catalog.jsonl` and, when present,
`examples.jsonl` beside it: context + plan prompts; without it plain prompts), the new one with its own `sft_render`
catalog and examples. On a CPU a request takes from seconds to a few minutes (measured on 4 cores: 2-72 s, set 3's
whole business processes the slowest), so a sample is used: `EVAL_PER_SET` (default 10) requests per set, the same
ones for both models, spread over each set and taken in turn from each set, at most `EVAL_MINUTES` (default 60) per
model; the comparison uses the requests both finished (scores of the installed
model are cached in `local_run/eval_cache.json`). With a GPU all 139 requests are used.
Better = more requests right, ties broken by more verified flows. Installing moves the old model (and its catalog /
examples) to `nanochat_base/old_model_<date>`, puts the new `model_<N>.pt` + `meta_<N>.json` in
`nanochat_base/chatsft_checkpoints/d8` and its `catalog.jsonl` + `examples.jsonl` in `rpa model/sft/`, where
`assistant/app.py` and `start_assistant.bat` read them. Restart the assistant to use it. History:
`nanochat_base/topup_history.jsonl`.

## Folders

```
rpa model/train_data/sft_render/      training data (from Drive or rendered here; feedback_train.jsonl added here)
rpa model/train_data/drive_model/     Drive's newest trained model + tokenizer (where a top-up starts)
rpa model/train_data/base/            the pretrained base model (full training)
rpa model/local_run/<mode>/           a training run: state.json, train.log, the last two chunks
rpa model/nanochat_base/              the installed model the assistant uses (+ old_model_<date> backups)
rpa model/sft/catalog.jsonl, examples.jsonl   the installed model's context (the model pack)
```

## Without the notebook (command line)

```
python training/local/data_pc.py  --rpa "<rpa model>" --drive-id <ID> --mode topup
python training/local/train_pc.py --rpa "<rpa model>" --mode topup --data "<rpa model>/train_data/sft_render" [--steps 300]
python training/local/eval_pc.py  --rpa "<rpa model>" --run-state "<rpa model>/local_run/topup/state.json" --install
```
`train_pc.py --measure 3` trains 3 steps and prints the seconds per step without keeping anything.
