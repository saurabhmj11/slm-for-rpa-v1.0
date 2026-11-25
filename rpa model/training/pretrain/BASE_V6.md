# The clean base model (v6)

The current base was pretrained on ClimbMix (CC BY-NC 4.0, non-commercial). v6 replaces it with a base pretrained
from zero on FineWeb-Edu (ODC-By 1.0; commercial use allowed with attribution) plus 15% of the owner's own domain
text, with a tokenizer trained on that mix. Data and licences: `DATA_CARD.md`. Code: `pretrain_data.py` (shards),
`base_setup.py` (profiles, estimates, the patch of nanochat's `base_train.py`, the tokenizer card, the pack),
`../local/pretrain_pc.py` (the PC driver, notebook MODE "pretrain") and Colab cells 3-4 with `PROFILE = "v6"`.

## The model and the run (profile "pc" / "t4", the same model)

| | v6 | current base |
|---|---|---|
| data | FineWeb-Edu sample-10BT (pinned revision) + 15% domain text | ClimbMix (non-commercial) |
| tokenizer | 16,384, trained on the mix (4.2 chars/token on flow JSON vs 2.1) | 16,384, web text only |
| model | d8: dim 512, 8 heads of 64, full attention, value embeddings, 75.5M parameters | the same |
| window | 2048 | 1024 |
| tokens | 1.51B = 20 per parameter (nanochat `--target-param-data-ratio 45` on its 33.6M scaling parameters) | 0.3B |
| steps | 5760 x 262,144 tokens (nanochat's own batch rule for this horizon) | 4577 x 65,536 |
| saved | every 200 steps on the PC (about 22 min), every 100 on Colab (about 12 min); the 2 newest kept | every 250 |

Why d8 and not d10 / d12: everything downstream (SFT, RL, eval, serving on a CPU, the timing of every mode) is
measured on d8; d10 costs 1.6x per token in SFT, RL and serving, d12 2.2x. v6 at d8 is a drop-in replacement of the
non-commercial base that sees 5x more text at twice the window. A larger model is a separate decision
(`--depth 10` works; the folder name stays `d8`, the size comes from the checkpoint's meta).

## Estimates (anchor: the owner's RTX 4060 Laptop 8 GB ran SFT at d8 / 4096 at 29.6k tokens/s, bf16, no torch.compile)

Other shapes scale by nanochat's FLOPs per token; the range adds 33% for small-matrix efficiency and laptop thermals.
T4: about 27k tokens/s at the anchor shape (Colab, fp16, torch.compile). Disk: shards + 3 checkpoints (2 kept + the
pack); `pretrain_pc.py --measure 30` measures the real speed and memory on the PC before the long run.

| depth | window | horizon | params | tokens | steps x batch | RTX 4060 h | T4 h | shards GB | disk GB | fits 8 GB? |
|---|---|---|---|---|---|---|---|---|---|---|
| d8 | 2048 | nanochat default (12) | 75.5M | 0.40B | 1536 x 256k | 2.9-3.8 | 3.1-4.2 | 0.74 | 3.2 | yes (2 x 2048) |
| d8 | 2048 | Chinchilla 20 x scaling params | 75.5M | 0.67B | 2560 x 256k | 4.8-6.3 | 5.2-6.9 | 1.23 | 3.6 | yes |
| **d8** | **2048** | **20 x all params (default)** | **75.5M** | **1.51B** | **5760 x 256k** | **10.7-14.2** | **11.7-15.6** | **2.77** | **5.2** | **yes** |
| d8 | 4096 | 20 x all params | 75.5M | 1.51B | 5760 x 256k | 14.2-19.0 | 15.6-20.8 | 2.77 | 5.2 | yes (1 x 4096, as SFT) |
| d10 | 2048 | nanochat default (12) | 122.6M | 0.72B | 1365 x 512k | 8.6-11.5 | 9.4-12.6 | 1.31 | 5.1 | likely (1 x 2048): measure |
| d10 | 2048 | Chinchilla 20 x scaling params | 122.6M | 1.19B | 2275 x 512k | 14.4-19.1 | 15.7-21.0 | 2.19 | 6.0 | likely: measure |
| d10 | 4096 | Chinchilla 20 x scaling params | 122.6M | 1.19B | 2275 x 512k | 18.7-25.0 | 20.5-27.4 | 2.19 | 6.0 | uncertain |
| d12 | 2048 | nanochat default (12) | 185.6M | 1.17B | 2232 x 512k | 22.2-29.6 | 24.3-32.4 | 2.15 | 7.8 | uncertain |
| d12 | 2048 | Chinchilla 20 x scaling params | 185.6M | 1.95B | 3720 x 512k | 37.0-49.3 | 40.6-54.1 | 3.58 | 9.2 | uncertain |

Plus about 10 min to build the data (network-bound) and a few minutes for the tokenizer. A T4 session is 2-5 h, so
the Colab run takes 3-6 sessions (it resumes); the PC is the better place for it.

## On the PC (notebook `ooa_local_pc.ipynb`)

1. Cell 1: `MODE = "pretrain"`, `PRETRAIN_MEASURE = 30`; run cells 1-5: it builds the data and the tokenizer and
   prints `MEASURED: ... tokens/s ... the full run takes about N h`.
2. Cell 1: `PRETRAIN_MEASURE = 0`; run cells 1-5. After any stop run cells 1-5 again (it resumes from the newest good
   checkpoint). It ends with `PRETRAINED: ...train_data/base_v6`.
3. Cell 1: `MODE = "full"`, `BASE = "v6"`; run cells 1-6 (SFT on v6, about 3.5-4.5 h; cell 6 installs it only if it
   beats the installed model).

## Safety

- NaN guard in `base_train`: a training loss that is not a number stops the run (bf16 at the first such step; fp16
  after 5 in a row, its GradScaler skips them), a validation that is not a number too, and every save checks the
  weights first: nothing broken is ever written. A rerun resumes from the newest complete, finite checkpoint.
- Nothing of the current base or the installed model is touched: the pack is `train_data/base_v6` (an earlier one
  is kept as `base_v6_before_<date>`), its SFT state `local_run/full_v6`, on Colab the folder `nanochat_base_v6`.
- A model and its tokenizer: every checkpoint trained from now records the sha1 of its tokenizer; the server, the
  assistant, `train_pc` / `rl_pc` / `eval_pc` refuse a pairing the model was not trained with (an old model with a v6
  tokenizer, or the reverse), and an install brings the model's tokenizer along.
- SmolTalk is off for SFT on v6 (`DATA_CARD.md`: it is largely another model's output, with that licence's naming and
  notice terms); set it only after a review.
