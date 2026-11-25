# Train the OOA model on a PC (Colab v4 recipe)

The easiest way is the notebook **`rpa model/ooa_local_pc.ipynb`** (open it in VS Code or Jupyter, run the cells
from top to bottom). It drives the three scripts in this folder, which reuse the Colab code (one source of truth:
`training/sft_setup.py`, `render_dataset.py`, `data_filter.py`, `ooa_format.py`):

| Script | What it does |
|---|---|
| `data_pc.py` | Downloads from Google Drive (`ooa-nanochat`, shared as "anyone with the link") into `rpa model/train_data/`: the rendered data `sft_render/` (when Drive has one that changed since the last download), else `sft/` (+ `sft_clean/`) and renders it here with the Colab cell 6 / 6d settings (data_filter, then render_dataset with the catalog, `--max-chars 13000 --max-tokens 4068 --answer-format compact --gold-dropout 0.5` and the new files long, business2, targeted, fix, clarify, beginner); the newest trained model (`chatsft_checkpoints/d8`, else the newest `chatsft_checkpoints_before_*`); the pretrained base model for full training. Drive files named `name (1).jsonl` are second uploads: reported and ignored. For a top-up it also renders the assistant's Accept / Edit feedback into `feedback_train.jsonl` in the same format (the full run trains on the Colab data only). |
| `train_pc.py` | Trains with nanochat's `scripts/chat_sft.py` from the pinned commit, patched by `sft_setup.patch_chat_sft` (the 4096 attention-window fix, the mixture planned for the step budget, rows longer than the window skipped, "Training data actually read" at the end), in saved chunks that resume after any stop. |
| `eval_pc.py` | Scores the new model and the installed one through the full assistant (`serving/ooa_server.Router`: catalog lock, repair, verify, 2 self-corrections) on sets 1-3 (`eval/plain_requests*.jsonl`) and the beginner set (`eval/beginner_requests.jsonl`), like Colab cell 8, and installs the new model only if it is better. |
| `setup_pc.py` | Notebook cell 2's first-run setup (standard library only): checks the Python (64-bit, 3.10+), copies `platform/`, `nanochat_base/`, `sft/`, `train_data/` and `assistant/assistant_data/` from an older `rpa model` folder next to this one (the code-only update zip has none of them), and puts nanochat at the pinned commit in `rpa model/nanochat`: from the older folder, else `git clone`, else GitHub's zip of the commit (Git is not needed). |
| `rl_pc.py` | MODE `rl` (NVIDIA GPU): reinforcement learning on the newest trained model, the Colab cell 9 recipe (`training/rl_setup.py`): nanochat's `scripts/chat_rl.py` from the pinned commit, patched and written as `scripts/chat_rl_ooa.py`; fresh requests (never in the training data, never close to a test prompt) with the server's prompts; the platform checker scores every expanded answer. Saved every 25 steps, resumes after any stop; the result goes through `eval_pc.py` like a top-up. |
| `pretrain_pc.py` | MODE `pretrain` (NVIDIA GPU): the clean base model (v6). Builds the data with `training/pretrain/pretrain_data.py` (FineWeb-Edu, ODC-By 1.0, streamed from Hugging Face + 15% the platform's own docs / code / catalog / rendered rows; no ClimbMix), trains the tokenizer (nanochat `tok_train`, 16,384 tokens, `tokenizer_card.json`), runs nanochat's `scripts/base_train.py` from the pinned commit patched by `training/pretrain/base_setup.py` (NaN guard, 2 newest checkpoints kept, the tokenizer's sha1 in every checkpoint), saved every 200 steps and resumed after any stop, and writes the pack `train_data/base_v6`. `--check` shows the plan and estimates, `--measure 30` the real speed. |
| `topup_nightly.bat` | Windows: the top-up every night at 01:30 (continues the run, then tests and installs). |
| `setup_wsl.sh` | Optional: a Python environment in WSL / Linux, where `torch.compile` works (faster full training on a GPU). |

## Modes (cell 1 `MODE`)

- **`auto`** (default): cell 3 picks and says why (`data_pc.choose_mode`): an NVIDIA GPU and a finished trained model
  on Drive (`chatsft_checkpoints/d8`, or a kept `chatsft_checkpoints_before_*`) or on this PC -> `refine`; a GPU and
  no trained model anywhere -> `full`; no GPU -> `topup`.
- **`refine`** (NVIDIA GPU; the step after a Colab run): continues the **finished Colab model** (v5; or this PC's own
  when it already grew from that Drive model, `train_pc.topup_start`) for `REFINE_STEPS` (1500, 1/8 of the Colab
  run) steps, saved every 250 steps (`REFINE_CHUNKS` 6, resumable), learning rate 0.35 (the Colab run peaked at 0.8
  and ended at 0: restarting at 0.8 would shake what it learned; 0.35 with its own warmdown lets the new rows settle). Every chunk passes Colab's base rates (embedding 0.3,
  unembedding 0.008, matrix 0.02, from the base model's meta when it is on the PC) explicitly: a saved SFT checkpoint
  stores them as None and chat_sft would otherwise fall back to unembedding 0.004.
  Mixture `sft_setup.REFINE_SHARES`: v5 14%, session 8% (the v5 model never saw it), beginner 20%, long 13%,
  business2 8%, targeted 6%, clarify 5%, fix 5%, feedback 3% when there is some; `REFINE_REPLAY` 18% replays every
  other file in the full mix's proportions. State in `local_run/refine/` (never shared with a top-up or full run).
  Estimated 40-65 minutes on a T4-class GPU (RTX 3060/4060; the Colab T4 with `torch.compile` trains 12000 steps in
  3.5-4.5 h, so 1500 steps take about 30-35 min there), plus 1-2 minutes of loading per chunk. Cell 6 then installs
  it only if it beats the Colab model it started from (and the model this PC had); if it does not, the Colab model is
  installed when it beats the PC's older one. `REFINE_THEN_RL = True` runs RL (MODE `rl`) and its test right after. If RL stops, run cell 6 again:
  it continues RL without testing the refine again (cell 5 skips the finished refine meanwhile).
- New files Colab's render lacks (e.g. `session_train.jsonl`, made after the v5 run): every mode but `full` takes them
  from Drive `sft/` when uploaded, else from the repo (`data/extra_sft/`), and renders only them here
  (`data_pc.render_extras`: Colab's settings, catalog and example index first; byte-identical to rendering them with
  the rest), instead of rendering everything again. `full` stays exactly the Colab data.
- **`full`** (NVIDIA GPU, explicit or `auto` without any trained model): exactly Colab cell 7, from the base model (it
  does not use the Colab model). 8000 steps of 32,768 tokens at the 4096 window from
  the **pretrained base model**, the full mixture planned for that budget (`sft_setup.plan_for_notebook`), 3000
  SmolTalk rows, learning rate 0.8 of the base one. The device batch follows the GPU memory (2 x 4096 on 14+ GB like
  the Colab T4, 1 below, 4 on 30+ GB); the tokens per step stay the same. About 2-3 hours on an RTX 3060/4060.
  Needs an RTX 20xx / GTX 16xx or newer. On plain Windows it runs without `torch.compile` (slower); WSL compiles.
  nanochat saves only at the end of a run, so by default (`FULL_CHUNKS = 1`) a stop means starting again; with
  `FULL_CHUNKS = 8` the run is saved every 1000 steps and resumes, with the same learning-rate schedule (each chunk
  continues the previous chunk's model and optimizer; flat lr over the first half, then the linear warmdown; the Muon
  momentum ramp counts the run's steps, not the chunk's) and the same planned mixture read with a new shuffle per chunk.
- **`pretrain`** (NVIDIA GPU): the clean base model (v6), see `training/pretrain/BASE_V6.md`: d8 at a 2048 window on
  1.51B tokens (20 per parameter), estimated 11-14 h on an RTX 4060 Laptop, about 5 GB of disk. State in
  `local_run/pretrain/`; the pack `train_data/base_v6` (tokenizer + card, `base_checkpoints/d8`, VERSION, data card).
  Never touches `train_data/base` or the installed model. `PRETRAIN_DATA = "drive"` uses shards someone uploaded to
  Drive `nanochat_base_v6/base_data_climbmix`; a v6 tokenizer on Drive (Colab PROFILE "v6") is used instead of training
  one, so both sides share it.
- **`full` with `BASE = "v6"`**: the same SFT run from the clean base, with its own tokenizer (its characters per token
  size the mixture), no SmolTalk rows (licence, see `training/pretrain/DATA_CARD.md`), state in `local_run/full_v6/`.
  Cell 6 installs it only if it scores better; the install brings its tokenizer along and keeps the old one.
  Every model trained from now on records the sha1 of its tokenizer; the assistant, the server and the training
  scripts refuse a model with a tokenizer it was not trained with (`model_backend.tokenizer_problem`).
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
- **`rl`** (NVIDIA GPU only): RL as Colab cell 9 on the newest trained model (always `train_pc.topup_start` now: after
  a refine that cell 6 installed, the refined model; an RL run in progress from an older start model is replaced by a
  new run, its result kept) (`RL_STEPS` (200) steps of 8 requests x 8
  answers, at most 1536 new tokens per answer, window 4096). Use it on a finished SFT model (v5), then cell 6 tests
  it and installs it only if it is better. Estimated about 3 hours per 100 steps on an RTX 3060/4060-class GPU (about a
  T4; range 2-6.5 h: sampling the answers token by token is most of the time, and requests whose answers all score the
  same are sampled but skipped). State in `local_run/rl/`.
- **`install`**: no training; tests Drive's newest model against the installed one and installs it if it is better
  (the quickest way to get a new Colab model onto the PC).

The old modes `long`, `curriculum` and `big` and the old plain-prompt `refine` (and their scripts `train_long.py`, `refine_local.py`,
`curriculum_local.py`, `big_local.py`, `train_local.py`, `topup_local.py`) are gone: they trained the old plain-prompt
format with plain-JSON answers and a 2048 window without the window fix, so their result would undo the v4 model.

## Broken weights (nan)

A run that goes unstable leaves weights that are not numbers (nan / inf). Nothing passes them on:
- `train_pc.py` (also `--check`, notebook cell 4) loads the start model's tensors and refuses a broken one by name,
  with the setting to use instead (cell 1 `REFINE_FROM = "pc"` / `"drive"`, `--start-from`); it never switches by
  itself. A run in progress whose saved chunk is broken (from an older version) starts over.
- The optimizer state saved with the start model is loaded only when it is finite (`optimizer_choice`; `--optimizer
  auto|load|fresh`); after a chunk went nan with it loaded, the next try starts with a fresh one.
- While a chunk trains, the step's own loss nan for 5 steps in a row or a nan validation stops it (the guard inside
  chat_sft from `sft_setup.patch_chat_sft`, exit code 86; train_pc `follow` also ends the chunk on a nan validation
  line). The loss on the step lines is a running average that stays nan after one bad step, so it is shown, never
  counted. Nothing of the chunk is saved, the state notes it (`nan_stops`), and a STOPPED line says what to send. A
  saved chunk is checked again before it counts.
- `eval_pc.py` never scores or installs a broken model (exit code 2); `rl_pc.py` refuses a broken start model (exit
  code 3: nothing started; notebook cell 6 then says rerunning will not help). `REFINE_THEN_RL` runs RL on this PC's
  model when `REFINE_FROM = "pc"`, else on the newest (`auto`).
- Colab cell 7 checks the model it saved: a broken one is moved to `chatsft_checkpoints_broken_<date>` (so the PC
  never downloads it as the newest), and cell 8 refuses to test one.

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
rpa model/train_data/base_v6/         the clean base model (MODE pretrain here, or Drive nanochat_base_v6)
rpa model/local_run/pretrain/base/    pretraining: shards (base_data_climbmix), tokenizer, the 2 newest checkpoints
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

## Windows notes

- The notebook has no `!` / `%` shell lines: every command is a Python list of arguments run with the kernel's own
  Python (`sys.executable`), and the output of every step is passed into the notebook (on Windows a notebook does not
  show what a child process writes to the console itself).
- Everything reads and writes UTF-8 explicitly (`encoding="utf-8"`, `PYTHONUTF8=1` for the scripts it starts):
  Windows' default code page (cp1252) cannot read the training data.
- No Git needed; no `torch.compile` on Windows (no compiler); training runs at below-normal priority and in a Windows
  job, so stopping a cell ends the whole process tree.
- Keep the folder outside OneDrive and at a short path (e.g. `C:\OOA\rpa model`): cell 2 warns otherwise.
