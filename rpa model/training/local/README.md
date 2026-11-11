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
