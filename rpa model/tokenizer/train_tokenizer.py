"""
Stage 2: train byte-level BPE tokenizers on the OOA corpus and measure compression.

Same algorithm as minbpe's RegexTokenizer, with one speed-up: chunks (regex "words") are
counted once and merges run over the unique chunks weighted by frequency, so training
on a few MB takes seconds instead of an hour.

Run from the "rpa model" folder:
    .venv\\Scripts\\python.exe tokenizer\\train_tokenizer.py
"""
import json
import os
import random
import time
from collections import Counter

import regex as re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "out")
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPLIT = re.compile(GPT4_PATTERN)
VOCAB_SIZES = [512, 1024, 2048, 4096]
N_HELD_OUT = 10


# ---- corpus -----------------------------------------------------------------
def load_flows():
    folder = os.path.join(RAW, "flows")
    names = sorted(n for n in os.listdir(folder) if n.endswith(".flow.json"))
    flows = {}
    for n in names:
        with open(os.path.join(folder, n), encoding="utf-8") as f:
            flows[n] = json.load(f)
    return flows


def minify(flow):
    return json.dumps(flow, separators=(",", ":"), ensure_ascii=False)


def pretty(flow):
    return json.dumps(flow, indent=2, ensure_ascii=False)


def load_docs():
    texts = []
    for root, _, files in os.walk(os.path.join(RAW, "docs")):
        for n in files:
            with open(os.path.join(root, n), encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
    return texts


# ---- BPE --------------------------------------------------------------------
def train_bpe(texts, vocab_size):
    chunk_counts = Counter()
    for t in texts:
        chunk_counts.update(SPLIT.findall(t))
    words = {c: list(c.encode("utf-8")) for c in chunk_counts}
    merges = {}                                   # (a, b) -> new id
    for new_id in range(256, vocab_size):
        pairs = Counter()
        for c, ids in words.items():
            w = chunk_counts[c]
            for p in zip(ids, ids[1:]):
                pairs[p] += w
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges[best] = new_id
        for c, ids in words.items():
            if len(ids) < 2:
                continue
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            words[c] = out
    return merges


def make_encoder(merges):
    cache = {}

    def encode_chunk(chunk):
        if chunk in cache:
            return cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: merges.get(p, float("inf")))
            if best not in merges:
                break
            new_id, out, i = merges[best], [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        cache[chunk] = ids
        return ids

    def encode(text):
        ids = []
        for chunk in SPLIT.findall(text):
            ids.extend(encode_chunk(chunk))
        return ids

    return encode


def vocab_bytes(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


# ---- main -------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    flows = load_flows()
    names = sorted(flows)
    held = set(random.sample(names, N_HELD_OUT))
    train_names = [n for n in names if n not in held]
    docs = load_docs()
    print(f"flows: {len(names)} ({len(train_names)} train, {len(held)} held out) | doc files: {len(docs)}")

    train_texts = [minify(flows[n]) for n in train_names] + docs
    held_min = [minify(flows[n]) for n in sorted(held)]
    held_pretty = [pretty(flows[n]) for n in sorted(held)]
    n_bytes_min = sum(len(t.encode()) for t in held_min)
    n_bytes_pretty = sum(len(t.encode()) for t in held_pretty)
    print(f"training text: {sum(len(t.encode()) for t in train_texts)/1e6:.2f} MB")
    print(f"held-out flows: {n_bytes_min:,} bytes minified, {n_bytes_pretty:,} bytes pretty-printed\n")

    os.makedirs(OUT, exist_ok=True)
    print(f"{'vocab':>6} {'train s':>8} | {'tokens (min)':>13} {'bytes/token':>12} | {'tokens (pretty)':>16}")
    print(f"{'bytes':>6} {'':>8} | {n_bytes_min:>13,} {1.0:>12.2f} | {n_bytes_pretty:>16,}")
    for vs in VOCAB_SIZES:
        t0 = time.time()
        merges = train_bpe(train_texts, vs)
        dt = time.time() - t0
        enc = make_encoder(merges)
        tm = sum(len(enc(t)) for t in held_min)
        tp = sum(len(enc(t)) for t in held_pretty)
        print(f"{vs:>6} {dt:>8.1f} | {tm:>13,} {n_bytes_min/tm:>12.2f} | {tp:>16,}")
        with open(os.path.join(OUT, f"ooa_bpe_{vs}.json"), "w", encoding="utf-8") as f:
            json.dump({"pattern": GPT4_PATTERN, "merges": [[a, b, i] for (a, b), i in merges.items()]}, f)
        if vs == VOCAB_SIZES[-1]:
            vocab = vocab_bytes(merges)
            print("\nlast 25 learned tokens (vocab 4096):")
            for i in range(vs - 25, vs):
                print(f"  {i}: {vocab[i].decode('utf-8', errors='replace')!r}")
            print("\nsample flow tokens (first held-out flow, vocab 4096):")
            ids = enc(held_min[0])[:40]
            print("  ", [vocab[i].decode('utf-8', errors='replace') for i in ids])

    # does the longest held-out flow fit a 2048-token window?
    lens = sorted(len(enc(t)) for t in held_min)
    print(f"\nheld-out flow lengths at vocab {VOCAB_SIZES[-1]} (minified): {lens}")
    print("fit in 2048 tokens:", sum(l <= 2048 for l in lens), "/", len(lens))
