"""
Logistics SLM — Memory benchmark.

Each probe is a multi-turn conversation where the correct assistant reply
depends on a fact stated exactly N turns earlier (N = 1..32). The model is
teacher-forced on the history, then greedily generates the probe reply.

Scoring (exact match): the generated reply up to <observation> (or the whole
reply when the answer comes from memory with no tool call) must equal the gold
reply. That covers the resolved entity in <thought>, the route, the tool and
its arguments, and for memory answers the recalled value itself.

Usage (from this folder):
    py data_gen_multiturn.py --bench --n 15     # once: build bench_memory.jsonl
    py context_bench.py --model ctx             # new recurrent-memory model
    py context_bench.py --model baseline        # original single-turn model
    py context_bench.py --model ctx --n 5       # quicker: 5 probes per cell
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "minbpe-master", "minbpe-master")))

BENCH_PATH = os.path.join(HERE, "bench_memory.jsonl")
CTX_BEST = os.path.join(HERE, "models", "ctx_best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_bench(n_per_cell=None, distances=None):
    with open(BENCH_PATH, encoding="utf-8") as f:
        items = [json.loads(x) for x in f]
    if distances:
        items = [it for it in items if it["distance"] in distances]
    if n_per_cell:
        seen, keep = defaultdict(int), []
        for it in items:
            k = (it["type"], it["distance"])
            if seen[k] < n_per_cell:
                seen[k] += 1
                keep.append(it)
        items = keep
    return items


def target_text(reply):
    """Part of the reply the model is responsible for (observations come from the tool)."""
    return reply.split("<observation>")[0].strip() if "<observation>" in reply else reply.strip()


def score(items, gen_fn, verbose=True):
    """gen_fn(history_text) -> generated reply text. Returns {(type, dist): (correct, n)} and failures."""
    res = defaultdict(lambda: [0, 0])
    fails = []
    t0 = time.time()
    for i, it in enumerate(items):
        pred = gen_fn(it["history"])
        ok = target_text(pred) == target_text(it["gold"])
        r = res[(it["type"], it["distance"])]
        r[0] += ok
        r[1] += 1
        if not ok and len(fails) < 8:
            fails.append((it["type"], it["distance"], target_text(it["gold"]), target_text(pred)))
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)")
    return res, fails


def report(res, fails=None):
    types = sorted({t for t, _ in res})
    dists = sorted({d for _, d in res})
    print(f"\n{'probe type':<14}" + "".join(f"{f'd={d}':>8}" for d in dists) + f"{'all':>8}")
    col = defaultdict(lambda: [0, 0])
    for t in types:
        row, tc, tn = "", 0, 0
        for d in dists:
            c, n = res.get((t, d), (0, 0))
            row += f"{100*c/n:>7.0f}%" if n else f"{'-':>8}"
            col[d][0] += c
            col[d][1] += n
            tc, tn = tc + c, tn + n
        print(f"{t:<14}{row}{100*tc/max(tn,1):>7.0f}%")
    tot_c = sum(c for c, _ in col.values())
    tot_n = sum(n for _, n in col.values())
    print(f"{'ALL':<14}" + "".join(f"{100*col[d][0]/max(col[d][1],1):>7.0f}%" for d in dists)
          + f"{100*tot_c/max(tot_n,1):>7.0f}%")
    if fails:
        print("\nSample failures:")
        for t, d, g, p in fails:
            print(f"  [{t} d={d}]\n    GOLD: {g}\n    PRED: {p}")
    return tot_c / max(tot_n, 1)


# ---- model adapters ----------------------------------------------------------
def ctx_generator(model, tok):
    stop = {tok.end_id, tok.obs_id}

    def gen(history):
        out = model.generate(tok.encode(history), max_new_tokens=80, stop_ids=stop)
        text = tok.decode([t for t in out if t != tok.end_id])
        return text + (" " if out and out[-1] == tok.obs_id else "")
    return gen


def load_ctx(path=CTX_BEST):
    from model_ctx import CtxGPT
    from tokenizer_ctx import CtxTokenizer
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = CtxGPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ctx_generator(model, CtxTokenizer())


def load_baseline():
    """Original single-turn model: sees only the last block_size tokens of the conversation."""
    from model import GPT
    from minbpe import RegexTokenizer
    ckpt = torch.load(os.path.join(HERE, "models", "logistics_best.pt"), map_location=DEVICE, weights_only=False)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = RegexTokenizer()
    tok.load(os.path.join(HERE, "models", "logistics_regex_bpe.model"))
    end_id, obs_id = tok.special_tokens["<end>"], tok.special_tokens["<observation>"]
    block = model.config.block_size

    @torch.no_grad()
    def gen(history):
        idx = torch.tensor([tok.encode(history, allowed_special="all")], device=DEVICE)
        out = []
        for _ in range(80):
            logits, _ = model(idx[:, -block:])
            nxt = int(logits[0, -1].argmax())
            if nxt == end_id:
                break
            out.append(nxt)
            if nxt == obs_id:
                break
            idx = torch.cat([idx, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        return tok.decode(out)
    return gen


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ctx", "baseline"], default="ctx")
    p.add_argument("--ckpt", default=CTX_BEST)
    p.add_argument("--n", type=int, default=None, help="probes per (type, distance) cell")
    a = p.parse_args()
    torch.set_num_threads(os.cpu_count())
    items = load_bench(a.n)
    gen = load_ctx(a.ckpt) if a.model == "ctx" else load_baseline()
    print(f"model={a.model} | {len(items)} probes")
    res, fails = score(items, gen)
    acc = report(res, fails)
    print(f"\noverall memory accuracy: {100*acc:.1f}%")
