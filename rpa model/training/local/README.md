# Train the OOA model on a laptop with an NVIDIA GPU

Same training as Colab cell 7 (2500 steps on the OOA data mix, including `natural_train.jsonl` and `core_train.jsonl` when present), in one
command, with no time limit or disconnects. It runs inside WSL (Ubuntu inside Windows) because nanochat compiles the
model with `torch.compile`, which plain Windows does not support.

Needs:
- Windows 10/11 and an NVIDIA **RTX 20xx / GTX 16xx or newer** GPU with 6 GB+ memory (8 GB+ better). Older cards
  (GTX 10xx, MX series) cannot run the training code.
- About 15 GB free disk.
- Internet for setup **and for the first training run** (it downloads a few hundred MB of general chat data once).

Expected time for one run: about 40-75 min on an RTX 3060/4060 class GPU (similar to the Colab T4).

## 1. Check the GPU (Windows Command Prompt)

```
nvidia-smi
```
It must show the card name and memory. If the command is not found, or the driver version is below 570, install the
latest driver from <https://www.nvidia.com/Download/index.aspx> and restart.

## 2. Install WSL (once, PowerShell as Administrator)

```
wsl --install
```
Restart Windows, open **Ubuntu** from the Start menu, choose a username and password.

## 3. Copy the files (Windows)

Unzip the **new** `ooa_assistant.zip` into `C:\ooa`. Check that these exist (an older zip does not have them):
`C:\ooa\rpa model\training\local\train_local.py` and `C:\ooa\rpa model\nanochat\scripts\chat_sft.py`.

From Google Drive `ooa-nanochat`, download into `C:\ooa_train` with exactly this layout:

```
C:\ooa_train\nanochat_base\tokenizer\tokenizer.pkl
C:\ooa_train\nanochat_base\tokenizer\token_bytes.pt
C:\ooa_train\nanochat_base\base_checkpoints\d8\model_<N>.pt        the HIGHEST step number in that Drive folder
C:\ooa_train\nanochat_base\base_checkpoints\d8\meta_<N>.json       same number
C:\ooa_train\nanochat_base\base_checkpoints\d8\optim_<N>_rank0.pt  same number
C:\ooa_train\sft\train_flows.jsonl, synth_train.jsonl, ...        the files of the Drive "sft" folder
```
Download the first five as single files (not whole folders). For `sft`, download the folder (Drive makes a zip) and
extract it so the files sit directly in `C:\ooa_train\sft\` (not in `C:\ooa_train\sft\sft\`).
`base_checkpoints` is the pretrained model every run starts from (not `chatsft_checkpoints`).

## 4. Set up Python with GPU support (Ubuntu window, once)

```
tr -d '\r' < "/mnt/c/ooa/rpa model/training/local/setup_wsl.sh" | bash
```
It ends with `GPU and compiler test: OK`. On a brand-new Ubuntu it may first wait a few minutes for Ubuntu's own updates.

## 5. Check, then train (Ubuntu window)

```
~/ooa-train-venv/bin/python "/mnt/c/ooa/rpa model/training/local/train_local.py" --work /mnt/c/ooa_train --check
```
Fix anything marked `!`, then start the real run:
```
~/ooa-train-venv/bin/python "/mnt/c/ooa/rpa model/training/local/train_local.py" --work /mnt/c/ooa_train
```
- GPU with less than 10 GB: add `--device-batch 2` (same result, a bit slower).
- Check an early line: `natural-wording file in the mix: True` (when `sft\natural_train.jsonl` exists).
- The run saves the model only at the end. Anything that stops it means starting again from the beginning, so:
  keep the Ubuntu window open; keep the laptop on power; set Sleep to Never (Settings > System > Power) and
  "When I close the lid" to Do nothing (Control Panel > Power Options); pause Windows Update for the day.
- If it stops with `Killed`, WSL ran out of memory (it gets half the laptop's RAM). Add `--smoltalk 0`, or give WSL
  more memory: create `C:\Users\<you>\.wslconfig` with the two lines `[wsl2]` and `memory=12GB`, run `wsl --shutdown`
  in Command Prompt, and open Ubuntu again.
- The log is saved in `C:\ooa_train`. If anything fails, send its last 30 lines.

## 6. Use the new model

The run ends by printing the new files, normally
`C:\ooa_train\nanochat_base\chatsft_checkpoints\d8\model_002500.pt` and `meta_002500.json`.
On the PC, in the assistant's `rpa model\nanochat_base\chatsft_checkpoints\d8\`, first move the old `model_*.pt` and
`meta_*.json` out (keep them as a backup), then copy the two new files in. The assistant loads the highest step number
in that folder, so an old file left there could be loaded instead. Restart `start_assistant.bat`, then score it:
```
python basic_test.py
```
A previous model in `C:\ooa_train\nanochat_base\chatsft_checkpoints` is never deleted: each run renames it to
`chatsft_checkpoints_before_<date>` first.

---

# Overnight top-up on a normal PC (no graphics card needed)

A full training run needs a GPU (Colab, Kaggle or an RTX laptop). A normal PC can still keep the model improving:
`topup_local.py` starts from the model the assistant uses now, trains a short round (200 steps, about 2 hours on
a 4-core CPU, faster with more cores) on your **Accept / Edit feedback** plus a sample of the existing training
data, then scores the old and the new model on the same 20 everyday requests. **The new model is installed only if
it scores better**; the old one is always kept in `nanochat_base\old_model_<date>`. History of every round:
`nanochat_base\topup_history.jsonl`.

## One-time setup (directly on Windows)

1. The assistant is installed and works (`setup_windows.bat`), e.g. in `F:\model slm 100 k parameter\rpa model v2\rpa model`.
2. Copy the training data: download the Drive folder `ooa-nanochat\sft` and put its files in `rpa model\sft\`
   (at least `natural_train.jsonl` and `core_train.jsonl`; `synth_train.jsonl`, `synth_pattern.jsonl`,
   `train_flows.jsonl`, `train_catalog.jsonl` help it not forget).
3. In Command Prompt, in the folder `rpa model\training\local`:
   ```
   topup_nightly.bat check
   ```
   It installs three small extra packages and must end with **READY**.

## Run it

- Once, now: double-click `topup_nightly.bat` (output goes to `topup_nightly.log` next to it).
- Every night at 01:30: `topup_nightly.bat install` once in Command Prompt (`topup_nightly.bat remove` stops it).
  The PC must be on and not asleep at that time.
- After a round that installed a new model, restart the assistant to use it.

(On Windows the model runs without `torch.compile`, which needs a C++ compiler Windows does not have; on a CPU that is
only a little slower: measured about 35-40 s per step on 4 cores, so a 200-step round takes about 2-2.5 hours. WSL also works: `setup_wsl.sh --cpu`, then run
`topup_local.py` with `~/ooa-train-venv/bin/python`.)

## How it learns from you

Use the assistant during the day and click **Accept** (the flow was right), **Edit** (fix it and save: the most
valuable signal) or **Reject**. Every round trains on all Accept/Edit answers that pass the platform's rules
(5x weight), so the requests you actually use get better first. Rejected requests are not trained on.

---

# Long run on a PC without a GPU (Colab-size training, day and night)

`train_long.py` does the full Colab training (2500 steps from the pretrained base model) on a CPU, split into 10
chunks of 250 steps that are each saved. A restart, update or sleep loses at most the chunk in progress: run it again
and it continues. It runs at low priority with one CPU core left free and 2 rows in memory at a time (about 4-5 GB
RAM; `--device-batch 1` about 3 GB), so the PC stays usable. About 1-1.5 days on 4 cores, less with more cores.
At the end the result is scored against the assistant's model and installed only if it is better.
Easiest: the notebook `ooa_local_pc.ipynb` with `MODE = "long"` in cell 1 (it downloads the base model).
