# Stage 1: char-level GPT on Tiny Shakespeare

Date: 2026-11-03. CPU only, Python 3.12 venv, nanoGPT (`nanoGPT-master`).

## Setup
- Data: 1,115,394 characters, vocab 65 (one token per character). 1,003,854 train / 111,540 val tokens.
- Model: 4 layers, 4 heads, 128-dim, context 64, 0.80M parameters, dropout 0.
- Training: 2000 iterations, batch 12, about 0.67 s per iteration on CPU (around 11 minutes total).
- Checkpoint: `training/out-shakespeare-char/ckpt.pt`. Full log: `stage1_train.log`.

## Loss curve
| step | train | val |
|---|---|---|
| 0 | 4.168 | 4.165 |
| 250 | 2.429 | 2.445 |
| 500 | 2.273 | 2.314 |
| 1000 | 1.971 | 2.053 |
| 1500 | 1.856 | 1.923 |
| 2000 | 1.765 | 1.886 |

## What it shows
- Step 0 loss 4.17 is about ln(65) = 4.17, i.e. random guessing over 65 characters. The model starts knowing nothing.
- Loss falls fast, then slowly. Val stays slightly above train, a small generalization gap that is still growing. More
  training would eventually overfit a model this small on 1M characters.
- Samples have Shakespeare's layout (speaker names, line breaks, punctuation) and many real words, but invented words
  and no meaning. A 0.8M model with 64 characters of context learns surface statistics, not sense.
- Same lesson as the 36K-parameter toy models in the parent folder: low loss means the model fits the data, not that it understands.

## To do next (exercises)
1. Read `nanoGPT-master/nanoGPT-master/model.py` and explain attention, the MLP and the loss in your own words.
2. Rerun with `--n_embd=256 --n_layer=6` and compare val loss and samples.
3. Rerun with `--block_size=128` and see if samples improve.
4. Plot the loss curve and note where overfitting starts.
