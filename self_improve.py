"""
self_improve.py � Praman Brain Mini DPO (Direct Preference Optimization)

This script implements a simplified Direct Preference Optimization (DPO) loop.
It trains the model to prefer syntactically correct JSON tool calls over 
corrupted/invalid ones.

Usage:
    python self_improve.py
"""
import time
import json
import random
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from tokenizer import AgentTokenizer
from data_gen import generate_dataset

# -- Config ----------------------------------------------------------------
BETA       = 0.1
BATCH_SIZE = 16
BLOCK_SIZE = 128
LR         = 5e-5
ITERS      = 500
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_IN    = "best_model_v2.pt"
CKPT_OUT   = "dpo_model_v2.pt"

torch.manual_seed(42)

def corrupt_json(text: str) -> str:
    """Introduce JSON syntax errors into a valid trajectory."""
    if "<tool_call>" not in text:
        return text
    parts = text.split("<tool_call>")
    head  = parts[0]
    tail  = parts[1]
    if "<observation>" not in tail:
        return text
    tc_parts = tail.split("<observation>")
    json_str = tc_parts[0].strip()
    obs_tail = tc_parts[1]
    corruption_type = random.choice(["drop_quote", "drop_brace", "wrong_colon"])
    if corruption_type == "drop_quote":
        json_str = json_str.replace('"', '', 1)
    elif corruption_type == "drop_brace":
        json_str = json_str.replace('}', '', 1)
    elif corruption_type == "wrong_colon":
        json_str = json_str.replace(':', '=', 1)
    return f"{head}<tool_call> {json_str} <observation>{obs_tail}"


def encode_and_pad(tok, text, block_size):
    """
    Encode text, truncate/pad to block_size.
    Returns (ids, resp_mask) where resp_mask has length (block_size - 1)
    and aligns with the *label* positions used in get_log_probs.
    """
    ids = tok.encode(text)

    # Find where the assistant response begins (at the <thought> token)
    thought_tok = tok.encode("<thought>")[0]
    try:
        resp_start = ids.index(thought_tok)
    except ValueError:
        resp_start = len(ids)   # no <thought> => treat whole thing as prompt

    # Truncate to block_size
    if len(ids) > block_size:
        ids = ids[:block_size]

    n = len(ids)

    # Build resp_mask over *label* positions (0 .. n-2).
    # label[i] = ids[i+1].  It is a response token if ids[i+1] is at or after resp_start.
    resp_mask = [
        1.0 if (i + 1) >= resp_start else 0.0
        for i in range(n - 1)
    ]

    # Pad ids to block_size with pad_id (0), not end_id
    pad_len = block_size - n
    ids = ids + [tok.pad_id] * pad_len

    # Pad resp_mask to (block_size - 1)
    resp_mask = resp_mask + [0.0] * (block_size - 1 - len(resp_mask))

    return ids, resp_mask


def build_dpo_batch(tok, n_examples):
    """Build a batch of preferred/rejected trajectories with correct response masks."""
    examples = generate_dataset(n_examples)

    pref_ids_list,  rej_ids_list  = [], []
    pref_mask_list, rej_mask_list = [], []

    for ex in examples:
        pid, pmask = encode_and_pad(tok, ex,               BLOCK_SIZE)
        rid, rmask = encode_and_pad(tok, corrupt_json(ex), BLOCK_SIZE)
        pref_ids_list.append(pid);   pref_mask_list.append(pmask)
        rej_ids_list.append(rid);    rej_mask_list.append(rmask)

    return (
        torch.tensor(pref_ids_list,  dtype=torch.long).to(DEVICE),
        torch.tensor(rej_ids_list,   dtype=torch.long).to(DEVICE),
        torch.tensor(pref_mask_list, dtype=torch.float32).to(DEVICE),
        torch.tensor(rej_mask_list,  dtype=torch.float32).to(DEVICE),
    )


def get_log_probs(model, x, resp_mask):
    """
    Compute sum of log-probs over response tokens only.
    x:         (B, T)     - padded token ids
    resp_mask: (B, T-1)   - 1 for response label positions, 0 for prompt/pad
    Returns:   (B,)       - summed log-probs over response tokens
    """
    inputs = x[:, :-1]           # (B, T-1)
    labels = x[:, 1:].clone()    # (B, T-1)

    logits, _ = model(inputs, targets=labels)   # (B, T-1, V)

    log_probs       = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)  # (B, T-1)

    return (token_log_probs * resp_mask).sum(-1)   # (B,)


def run_dpo():
    print("Loading tokenizer and models...")
    tok = AgentTokenizer()

    ckpt         = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
    cfg          = ckpt["config"]

    # Policy model (trained)
    policy_model = GPT(cfg).to(DEVICE)
    policy_model.load_state_dict(ckpt["model"])
    policy_model.train()

    # Reference model (frozen)
    ref_model = GPT(cfg).to(DEVICE)
    ref_model.load_state_dict(ckpt["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

    print(f"Starting DPO training for {ITERS} iterations...")
    t0 = time.time()

    for it in range(ITERS):
        x_pref, x_rej, p_mask, r_mask = build_dpo_batch(tok, BATCH_SIZE)

        with torch.no_grad():
            ref_pref_logp = get_log_probs(ref_model, x_pref, p_mask)
            ref_rej_logp  = get_log_probs(ref_model, x_rej,  r_mask)

        policy_pref_logp = get_log_probs(policy_model, x_pref, p_mask)
        policy_rej_logp  = get_log_probs(policy_model, x_rej,  r_mask)

        ratio_pref = policy_pref_logp - ref_pref_logp
        ratio_rej  = policy_rej_logp  - ref_rej_logp
        loss = -F.logsigmoid(BETA * (ratio_pref - ratio_rej)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        if it % 50 == 0 or it == ITERS - 1:
            pref_r = (BETA * ratio_pref).mean().item()
            rej_r  = (BETA * ratio_rej).mean().item()
            print(f"Iter {it:4d} | Loss: {loss.item():.4f} | "
                  f"Pref: {pref_r:.3f} | Rej: {rej_r:.3f} | Gap: {pref_r - rej_r:.3f}")

    torch.save({"model": policy_model.state_dict(), "config": cfg}, CKPT_OUT)
    print(f"DPO Complete. Saved to {CKPT_OUT}. ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    run_dpo()
