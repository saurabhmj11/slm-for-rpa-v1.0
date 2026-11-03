"""
Train the ~100K-param agent-brain prototype on synthetic agent
trajectories (routing + tool-call + reasoning), all in-memory since the
dataset is tiny by design at this scale.
"""
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from tokenizer import AgentTokenizer
from data_gen import generate_dataset

import argparse
import os
import time

# ---- config ----
N_EXAMPLES = 20000
BLOCK_SIZE = 128
BATCH_SIZE = 8
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "ckpt_v2.pt"
BEST_PATH = "best_model_v2.pt"

torch.manual_seed(42)


def build_split(tok, n_examples, block_size):
    """Encode every example, pad/pack into fixed block_size rows."""
    examples = generate_dataset(n_examples)
    rows = []
    for ex in examples:
        ids = tok.encode(ex)
        if len(ids) > block_size:
            ids = ids[:block_size]
        ids = ids + [tok.pad_id] * (block_size - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def get_batch(data, batch_size, device, pad_id):
    ix = torch.randint(0, data.size(0), (batch_size,))
    batch = data[ix]
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].clone()
    y[y == pad_id] = -1  # ignore padding in the loss
    y = y.to(device)
    return x, y


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
    args = parser.parse_args()

    tok = AgentTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE - 1)
    model = GPT(cfg).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters)

    start_iter, best_val = 0, float("inf")
    if os.path.exists(CKPT_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_iter = ckpt["iter"]
            best_val = ckpt["best_val"]
            print(f"resumed from {CKPT_PATH} at iter {start_iter}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting fresh.")

    print(f"device: {DEVICE} | params: {model.num_params():,}")
    print("generating synthetic train/val sets...")
    train_data = build_split(tok, N_EXAMPLES, BLOCK_SIZE)
    val_data = build_split(tok, max(1000, N_EXAMPLES // 10), BLOCK_SIZE)

    t0 = time.time()
    end_iter = start_iter + args.iters
    for it in range(start_iter, end_iter):
        if it % EVAL_INTERVAL == 0 or it == args.total_iters - 1:
            tr = estimate_loss(model, train_data, BATCH_SIZE, DEVICE, EVAL_ITERS, tok.pad_id)
            va = estimate_loss(model, val_data, BATCH_SIZE, DEVICE, EVAL_ITERS, tok.pad_id)
            print(f"iter {it:5d} | train loss {tr:.4f} | val loss {va:.4f} | lr {optimizer.param_groups[0]['lr']:.5f} | {time.time()-t0:.1f}s")
            if va < best_val:
                best_val = va
                torch.save({"model": model.state_dict(), "config": cfg}, BEST_PATH)

        x, y = get_batch(train_data, BATCH_SIZE, DEVICE, tok.pad_id)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "iter": end_iter, "best_val": best_val,
    }, CKPT_PATH)
    print(f"stopped at iter {end_iter}/{args.total_iters}. best val loss so far: {best_val:.4f}. ({time.time()-t0:.1f}s this run)")
