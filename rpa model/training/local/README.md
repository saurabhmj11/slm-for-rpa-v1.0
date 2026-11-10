# Train the OOA model on a laptop with an NVIDIA GPU

Same training as Colab cell 7 (2000 steps on the OOA data mix, including `natural_train.jsonl` when present), in one
command, with no time limit or disconnects. It runs inside WSL (Ubuntu inside Windows) because nanochat compiles the
model with `torch.compile`, which plain Windows does not support.

Needs: Windows 10/11, an NVIDIA GPU with 6 GB+ memory (8 GB+ better), about 15 GB free disk, internet for setup.
Expected time for one run: about 30-60 min on an RTX 3060/4060 class GPU (similar to the Colab T4).

## 1. Check the GPU (Windows Command Prompt)

```
nvidia-smi
```
It must show the card name and memory. If the command is not found, install the latest driver from
<https://www.nvidia.com/Download/index.aspx> and restart.

## 2. Install WSL (once, PowerShell as Administrator)

```
wsl --install
```
Restart Windows, open **Ubuntu** from the Start menu, choose a username and password.

## 3. Copy the files (Windows)

Unzip `ooa_assistant.zip` into `C:\ooa` (you get `C:\ooa\rpa model\...`).

From Google Drive `ooa-nanochat`, download into `C:\ooa_train` keeping this layout (download single files, not
whole folders, except `sft`):

```
C:\ooa_train\nanochat_base\tokenizer\tokenizer.pkl
C:\ooa_train\nanochat_base\tokenizer\token_bytes.pt
C:\ooa_train\nanochat_base\base_checkpoints\d8\model_<N>.pt        the HIGHEST step number in that Drive folder
C:\ooa_train\nanochat_base\base_checkpoints\d8\meta_<N>.json       same number
C:\ooa_train\nanochat_base\base_checkpoints\d8\optim_<N>_rank0.pt  same number
C:\ooa_train\sft\...                                               the whole Drive "sft" folder (unzip it here)
```
`base_checkpoints` is the pretrained model every training run starts from (not `chatsft_checkpoints`).

## 4. Set up Python with GPU support (Ubuntu window, once)

```
bash "/mnt/c/ooa/rpa model/training/local/setup_wsl.sh"
```
It ends with `GPU visible: True`.

## 5. Check, then train (Ubuntu window)

```
~/ooa-train-venv/bin/python "/mnt/c/ooa/rpa model/training/local/train_local.py" --work /mnt/c/ooa_train --check
```
Fix anything marked `!`, then start the real run:
```
~/ooa-train-venv/bin/python "/mnt/c/ooa/rpa model/training/local/train_local.py" --work /mnt/c/ooa_train
```
GPU with less than 10 GB: add `--device-batch 2` (same result, a bit slower). Keep the laptop on power and stop it
from sleeping (Settings > System > Power > Sleep: Never) while it trains. The log is saved in `C:\ooa_train`.

Check the first lines: `natural-wording file in the mix: True` (when `sft\natural_train.jsonl` exists).

## 6. Use the new model

The new model is `C:\ooa_train\nanochat_base\chatsft_checkpoints\d8\model_002000.pt` and `meta_002000.json`.
Copy those two files into the assistant's `rpa model\nanochat_base\chatsft_checkpoints\d8\` (replace the old ones),
restart `start_assistant.bat`, then score it:
```
python basic_test.py
```
A previous model in `chatsft_checkpoints` is never deleted: each run renames it to
`chatsft_checkpoints_before_<date>` first.
