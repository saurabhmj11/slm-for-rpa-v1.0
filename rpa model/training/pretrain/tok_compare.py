"""Compare nanochat tokenizers on our text: characters per token by kind of text, tokens per action id, and the
special tokens the chat format needs.

    python tok_compare.py --tok current=DIR --tok d16k=DIR ... --render SFT_RENDER [--catalog CAT] [--web-dir D]

Every DIR is a nanochat tokenizer folder (tokenizer.pkl from scripts/tok_train.py). The text is held out from
tokenizer training: rendered val.jsonl rows (request / context / plan / flow JSON), the eval/*.jsonl prompts (plain
English office requests) and FineWeb-Edu's validation row groups (the ones pretrain_data.py never trains on).
"""
import argparse
import json
import pickle
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pretrain_data as P                                    # noqa: E402

# nanochat/tokenizer.py SPECIAL_TOKENS at the pinned commit: names and order must not change
SPECIAL = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
           "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]


def load(d):
    with open(Path(d) / "tokenizer.pkl", "rb") as f:
        return pickle.load(f)                                 # a tiktoken Encoding


def specials_ok(enc):
    """(ok, detail): every chat special token present, in nanochat's order, as the last ids of the vocabulary."""
    st = enc._special_tokens
    ids = [st.get(s) for s in SPECIAL]
    n = enc.n_vocab
    ok = None not in ids and ids == list(range(n - len(SPECIAL), n)) and set(st) == set(SPECIAL)
    return ok, dict(zip(SPECIAL, ids))


def texts(render_dirs, eval_dir, catalog, web_dir, web_groups):
    rows = []
    for d in render_dirs:
        for f in sorted(Path(d).glob("val*.jsonl")):
            rows += list(P._jsonl(f))
    out = {"request": [], "context": [], "plan": [], "flow": [], "conversation": []}
    for r in rows:
        m = r.get("messages", [])
        u = next((x["content"] for x in m if x.get("role") == "user"), "")
        a = next((x["content"] for x in reversed(m) if x.get("role") == "assistant"), "")
        req, _, ctx = u.partition("### Context")
        plan, _, flow = a.partition("Flow:\n")
        out["request"].append(req)
        out["context"].append(ctx)
        out["conversation"].append(u + a)
        if flow:
            out["plan"].append(plan)
            out["flow"].append(flow)
    out["eval prompts"] = P.eval_prompts(eval_dir)
    if web_groups:
        w = P.WebSource(web_dir, val_groups=web_groups)
        out["web (FineWeb-Edu val)"] = [t for u in w.val_units() for t in w.texts(u)]
    ids = []
    if catalog:
        for l in open(catalog, encoding="utf-8"):
            try:
                ids.append(json.loads(l)["id"])
            except (ValueError, KeyError):
                pass
    return out, ids


def measure(enc, out, ids):
    res = {}
    for k, ts in out.items():
        c = sum(map(len, ts))
        t = sum(len(enc.encode_ordinary(x)) for x in ts)
        res[k] = (c / max(1, t), t)
    bare = [len(enc.encode_ordinary(i)) for i in ids]
    # as the flow writes it; the shared '{"action":"' prefix and '"}' are subtracted
    base = len(enc.encode_ordinary('{"action":""}'))
    inflow = [len(enc.encode_ordinary('{"action":"%s"}' % i)) - base for i in ids]
    res["id"] = (statistics.mean(bare) if ids else 0, max(bare) if ids else 0,
                 statistics.mean(inflow) if ids else 0)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tok", action="append", required=True, help="NAME=tokenizer folder; the first is the baseline")
    ap.add_argument("--render", action="append", default=[], help="rendered data folder with val.jsonl; repeatable")
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--web-dir", default=None, help="local FineWeb-Edu parquet files (default: stream the pinned ones)")
    ap.add_argument("--web-groups", type=int, default=2, help="FineWeb validation row groups to measure (0 = none)")
    ap.add_argument("--json", default=None, help="also write the numbers here")
    a = ap.parse_args(argv)
    out, ids = texts(a.render, a.eval_dir, a.catalog, a.web_dir, a.web_groups)
    print("held-out text: " + ", ".join(f"{k} {len(v)} docs / {sum(map(len, v)) / 1e6:.2f}M chars"
                                         for k, v in out.items()) + f"; {len(ids)} action ids")
    kinds = list(out)
    rows, base = [], None
    for spec in a.tok:
        name, d = spec.split("=", 1)
        enc = load(d)
        ok, sp = specials_ok(enc)
        r = measure(enc, out, ids)
        if base is None:
            base = r
        rows.append({"name": name, "vocab": enc.n_vocab, "specials_ok": ok, "specials": sp,
                     **{k: round(r[k][0], 3) for k in kinds},
                     "tokens": {k: r[k][1] for k in kinds},
                     "flow_tokens_vs_first": round(r["flow"][1] / max(1, base["flow"][1]) - 1, 3),
                     "conversation_tokens_vs_first": round(r["conversation"][1] / max(1, base["conversation"][1]) - 1, 3),
                     "id_tokens_mean": round(r["id"][0], 2), "id_tokens_max": r["id"][1],
                     "id_tokens_in_flow": round(r["id"][2], 2)})
    hdr = ["tokenizer", "vocab"] + [f"{k} c/t" for k in kinds] + ["flow tok vs 1st", "conv tok vs 1st",
                                                                     "id tok (bare)", "id tok max", "id tok (in flow)",
                                                                     "specials"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join([r["name"], f"{r['vocab']:,}"] + [f"{r[k]:.2f}" for k in kinds] +
                                [f"{100 * r['flow_tokens_vs_first']:+.1f}%", f"{100 * r['conversation_tokens_vs_first']:+.1f}%",
                                 f"{r['id_tokens_mean']:.2f}", str(r["id_tokens_max"]), f"{r['id_tokens_in_flow']:.2f}",
                                 "same" if r["specials_ok"] else "DIFFERENT"]) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    main()
