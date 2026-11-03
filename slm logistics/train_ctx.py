"""
Logistics SLM — Memory-augmented model training.

Trains model_ctx.py (with recurrent memory slots) on the multi-turn dataset.
Uses a short-to-long curriculum:
- Starts with short sequences (max 2 segments = 1024 tokens)
- Progresses to longer sequences as loss drops
- Loss is computed *only* on the assistant tokens (using the tokenizer mask)

Usage:
    $env:PYTHONPATH="..;..\\minbpe-master\\minbpe-master"; py train_ctx.py
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "minbpe-master", "minbpe-master")))
from model_ctx import CtxGPT, CtxConfig  # noqa: E402
from tokenizer_ctx import CtxTokenizer  # noqa: E402

# ---- config ----
HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(HERE, "models", "ctx_bpe.model")
TRAIN_PATH = os.path.join(HERE, "conv_train.jsonl")
VAL_PATH = os.path.join(HERE, "conv_val.jsonl")

SEG_LEN = 512
N_MEM = 16
BATCH_SIZE = 12
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 40
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = os.path.join(HERE, "models", "ctx_ckpt.pt")
BEST_PATH = os.path.join(HERE, "models", "ctx_best.pt")

torch.manual_seed(42)


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def get_batch(data, tok, batch_size, device, max_segments=None):
    """
    Returns (x, y). x: (B, N), y: (B, N).
    y is -1 where loss should be ignored (user turns, tool obs, padding).
    """
    # Sample and encode
    texts = random.sample(data, batch_size)
    encoded = [tok.encode_with_mask(t) for t in texts]
    
    # Truncate to max_segments if needed
    max_tokens = max_segments * SEG_LEN if max_segments else None
    
    # Find max length in this batch (rounded up to nearest segment)
    max_len = max(len(ids) for ids, _ in encoded)
    if max_tokens and max_len > max_tokens:
        max_len = max_tokens
    
    n_seg = (max_len + SEG_LEN - 1) // SEG_LEN
    pad_len = n_seg * SEG_LEN
    
    batch_x, batch_y = [], []
    for ids, mask in encoded:
        if max_tokens and len(ids) > max_tokens:
            ids = ids[:max_tokens]
            mask = mask[:max_tokens]
            
        pad = [tok.pad_id] * (pad_len - len(ids))
        x = ids + pad
        
        # Shift mask right for prediction: y[t] = label for predicting x[t+1]
        y_ids = (ids[1:] + [tok.pad_id]) + pad
        y_mask = (mask[1:] + [0]) + [0] * len(pad)
        y = [yi if m else -1 for yi, m in zip(y_ids, y_mask)]
        
        batch_x.append(x)
        batch_y.append(y)
        
    return torch.tensor(batch_x, device=device), torch.tensor(batch_y, device=device)


@torch.no_grad()
def estimate_loss(model, data, tok, batch_size, device, iters, max_segments):
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, tok, batch_size, device, max_segments)
        _, loss = model(x, y)
        losses[i] = loss.item() if loss is not None else 0.0
    model.train()
    return losses.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=MAX_ITERS)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    # ---- tokenizer ----
    tok = CtxTokenizer(TOKENIZER_PATH)

    # ---- data ----
    train_ex = load_dataset(TRAIN_PATH)
    val_ex = load_dataset(VAL_PATH)
    print(f"train: {len(train_ex):,} | val: {len(val_ex):,} | vocab: {tok.vocab_size}")

    # ---- model ----
    cfg = CtxConfig(
        vocab_size=tok.vocab_size,
        seg_len=SEG_LEN,
        n_mem=N_MEM,
    )
    model = CtxGPT(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)

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
    
    # Curriculum: start with short sequences, unlock longer ones as training progresses
    curriculum = [
        (0.0, 2),   # start with 2 segments
        (0.3, 4),   # at 30% through training, unlock 4 segments
        (0.6, 8),   # at 60% through training, unlock 8 segments
        (0.8, None) # at 80%, full sequences
    ]
    
    for it in range(start_iter, args.iters):
        # Determine max segments for curriculum
        progress = it / args.iters
        max_seg = None
        for p, s in curriculum:
            if progress >= p:
                max_seg = s
                
        if it % EVAL_INTERVAL == 0 or it == args.iters - 1:
            tr = estimate_loss(model, train_ex, tok, BATCH_SIZE, DEVICE, EVAL_ITERS, max_seg)
            va = estimate_loss(model, val_ex, tok, BATCH_SIZE, DEVICE, EVAL_ITERS, max_seg)
            print(f"iter {it:5d} | train {tr:.4f} | val {va:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.5f} | seg_limit {max_seg} | {time.time()-t0:.1f}s")
            
            # We want to checkpoint based on long-context val loss, so we estimate it directly
            long_va = estimate_loss(model, val_ex, tok, BATCH_SIZE // 2 or 1, DEVICE, EVAL_ITERS, None)
            
            if long_va < best_val:
                best_val = long_va
                print(f"  new best long-val loss: {best_val:.4f} -> saving {BEST_PATH}")
                torch.save({"model": model.state_dict(), "config": cfg,
                            "tokenizer": TOKENIZER_PATH}, BEST_PATH)

        x, y = get_batch(train_ex, tok, BATCH_SIZE, DEVICE, max_seg)
        _, loss = model(x, y)
        
        if loss is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "iter": args.iters, "best_val": best_val,
    }, CKPT_PATH)
    print(f"stopped at iter {args.iters}. best long val loss: {best_val:.4f} "
          f"-> {BEST_PATH} ({time.time()-t0:.1f}s this run)")
