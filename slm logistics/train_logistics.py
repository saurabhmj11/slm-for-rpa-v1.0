"""
Logistics SLM — Layer 2: Training.

Trains a ~100K-param LLaMA-style model (../model.py) on the logistics
agent trajectories, using the fixed 90/10 split produced by
tokenizer_logistics.py (train.txt / val.txt) and the RegexTokenizer
saved at models/logistics_regex_bpe.model.

Run from this folder:
    $env:PYTHONPATH="..;..\\minbpe-master\\minbpe-master"; py train_logistics.py
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import GPT, GPTConfig  # noqa: E402
from minbpe import RegexTokenizer  # noqa: E402

# ---- config ----
TOKENIZER_PATH = os.path.join("models", "logistics_regex_bpe.model")
TRAIN_PATH = "train.txt"
VAL_PATH = "val.txt"
END_TOKEN = "<end>"

N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
BATCH_SIZE = 32
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = os.path.join("models", "logistics_ckpt.pt")
BEST_PATH = os.path.join("models", "logistics_best.pt")

torch.manual_seed(42)


def load_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return [ex.strip() + END_TOKEN for ex in raw.split(END_TOKEN) if ex.strip()]


def encode_split(tok, examples, block_size, pad_id):
    """Encode each example and right-pad to block_size (one example per row)."""
    rows, truncated = [], 0
    for ex in examples:
        ids = tok.encode(ex, allowed_special="all")
        if len(ids) > block_size:
            ids = ids[:block_size]
            truncated += 1
        rows.append(ids + [pad_id] * (block_size - len(ids)))
    if truncated:
        print(f"  warning: {truncated} examples truncated to {block_size} tokens")
    return torch.tensor(rows, dtype=torch.long)


def get_batch(data, batch_size, device, pad_id):
    ix = torch.randint(0, data.size(0), (batch_size,))
    batch = data[ix]
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].clone()
    y[y == pad_id] = -1  # ignore padding in the loss
    return x, y.to(device)


@torch.no_grad()
def estimate_loss(model, data, batch_size, device, iters, pad_id):
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, batch_size, device, pad_id)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=MAX_ITERS, help="iterations to run THIS invocation")
    parser.add_argument("--total_iters", type=int, default=MAX_ITERS, help="total iters for the cosine LR schedule")
    parser.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    args = parser.parse_args()

    # ---- tokenizer ----
    tok = RegexTokenizer()
    tok.load(TOKENIZER_PATH)
    pad_id = tok.special_tokens["<pad>"]
    vocab_size = max(max(tok.vocab), max(tok.special_tokens.values())) + 1

    # ---- data ----
    train_ex = load_examples(TRAIN_PATH)
    val_ex = load_examples(VAL_PATH)
    max_len = max(len(tok.encode(ex, allowed_special="all")) for ex in train_ex[:2000] + val_ex)
    block_size = max_len + 1  # +1 because x/y are shifted by one
    print(f"train: {len(train_ex):,} | val: {len(val_ex):,} | vocab: {vocab_size} | block_size: {block_size}")

    print("encoding splits...")
    train_data = encode_split(tok, train_ex, block_size, pad_id)
    val_data = encode_split(tok, val_ex, block_size, pad_id)

    # ---- model ----
    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=block_size - 1,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        dropout=0.1, share_layers=True,
    )
    model = GPT(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters)

    start_iter, best_val = 0, float("inf")
    if os.path.exists(CKPT_PATH) and not args.fresh:
        try:
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_iter, best_val = ckpt["iter"], ckpt["best_val"]
            print(f"resumed from {CKPT_PATH} at iter {start_iter}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting fresh.")

    print(f"device: {DEVICE} | params: {model.num_params():,}")

    # ---- train ----
    t0 = time.time()
    end_iter = min(start_iter + args.iters, args.total_iters)
    for it in range(start_iter, end_iter):
        if it % EVAL_INTERVAL == 0 or it == end_iter - 1:
            tr = estimate_loss(model, train_data, BATCH_SIZE, DEVICE, EVAL_ITERS, pad_id)
            va = estimate_loss(model, val_data, BATCH_SIZE, DEVICE, EVAL_ITERS, pad_id)
            print(f"iter {it:5d} | train {tr:.4f} | val {va:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.5f} | {time.time()-t0:.1f}s")
            if va < best_val:
                best_val = va
                torch.save({"model": model.state_dict(), "config": cfg,
                            "tokenizer": TOKENIZER_PATH}, BEST_PATH)

        x, y = get_batch(train_data, BATCH_SIZE, DEVICE, pad_id)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    os.makedirs("models", exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "iter": end_iter, "best_val": best_val,
    }, CKPT_PATH)
    print(f"stopped at iter {end_iter}/{args.total_iters}. best val loss: {best_val:.4f} "
          f"-> {BEST_PATH} ({time.time()-t0:.1f}s this run)")
