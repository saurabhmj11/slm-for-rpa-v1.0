"""
Logistics SLM — Layer 3: Evaluation + interactive inference.

Feeds each val.txt user prompt to the trained model, greedily generates the
assistant turn, and checks routing (<route>), tool choice (<tool_call>) and
exact-match of the full response against the ground truth.

Usage (from this folder):
    $env:PYTHONPATH="..;..\\minbpe-master\\minbpe-master"
    py evaluate_logistics.py              # evaluate on val.txt
    py evaluate_logistics.py --n 300      # evaluate on first 300 val examples
    py evaluate_logistics.py --chat       # interactive prompt
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import GPT  # noqa: E402
from minbpe import RegexTokenizer  # noqa: E402

BEST_PATH = os.path.join("models", "logistics_best.pt")
TOKENIZER_PATH = os.path.join("models", "logistics_regex_bpe.model")
VAL_PATH = "val.txt"
END_TOKEN = "<end>"
ASSISTANT_PREFIX = "<|im_start|> assistant\n"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = RegexTokenizer()
    tok.load(TOKENIZER_PATH)
    return model, tok


def build_prompt(user_msg):
    return f"<|im_start|> user\n{user_msg}<|im_end|>\n{ASSISTANT_PREFIX}"


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=100):
    end_id = tok.special_tokens[END_TOKEN]
    ids = tok.encode(prompt, allowed_special="all")
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    block = model.config.block_size
    out = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        nxt = int(logits[0, -1].argmax())
        if nxt == end_id:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return tok.decode(out).strip()


def field(text, tag):
    m = re.search(rf"<{tag}>\s*(\S+)", text)
    return m.group(1) if m else None


def parse_example(ex):
    user = ex.split("<|im_start|> user\n", 1)[1].split("<|im_end|>", 1)[0]
    asst = ex.split(ASSISTANT_PREFIX, 1)[1].replace(END_TOKEN, "").strip()
    return user, asst


def evaluate(model, tok, n):
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    examples = [e.strip() + END_TOKEN for e in raw.split(END_TOKEN) if e.strip()][:n]

    stats = defaultdict(lambda: {"n": 0, "route": 0, "tool": 0, "exact": 0})
    failures = []
    for i, ex in enumerate(examples):
        user, gold = parse_example(ex)
        pred = generate(model, tok, build_prompt(user))
        agent = field(gold, "route")
        s = stats[agent]
        s["n"] += 1
        r_ok = field(pred, "route") == agent
        t_ok = field(pred, "tool_call") == field(gold, "tool_call")
        s["route"] += r_ok
        s["tool"] += t_ok
        s["exact"] += pred == gold
        if not (r_ok and t_ok) and len(failures) < 5:
            failures.append((user, gold, pred))
        if (i + 1) % 100 == 0:
            print(f"  evaluated {i+1}/{len(examples)}")

    print(f"\n{'agent':<18}{'n':>5}{'route%':>9}{'tool%':>8}{'exact%':>9}")
    tot = {"n": 0, "route": 0, "tool": 0, "exact": 0}
    for agent in sorted(stats, key=str):
        s = stats[agent]
        for k in tot:
            tot[k] += s[k]
        print(f"{str(agent):<18}{s['n']:>5}{100*s['route']/s['n']:>8.1f}%"
              f"{100*s['tool']/s['n']:>7.1f}%{100*s['exact']/s['n']:>8.1f}%")
    print(f"{'TOTAL':<18}{tot['n']:>5}{100*tot['route']/tot['n']:>8.1f}%"
          f"{100*tot['tool']/tot['n']:>7.1f}%{100*tot['exact']/tot['n']:>8.1f}%")

    if failures:
        print("\nSample routing/tool failures:")
        for user, gold, pred in failures:
            print(f"  Q:    {user}\n  GOLD: {gold}\n  PRED: {pred}\n")


def chat(model, tok):
    print("Logistics SLM — type a question (empty line to quit)")
    while True:
        q = input("> ").strip()
        if not q:
            break
        print(generate(model, tok, build_prompt(q)), "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500, help="number of val examples to evaluate")
    p.add_argument("--chat", action="store_true")
    a = p.parse_args()
    m, t = load_model()
    chat(m, t) if a.chat else evaluate(m, t, a.n)
