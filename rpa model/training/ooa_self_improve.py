"""Self-improvement round for the OOA model (expert iteration with a verifier). Copy to nanochat/scripts/.

Each round:
  1. make FRESH requests (new generator seed) plus replay the requests it failed on before
  2. the current model writes K candidate flows per request
  3. the verifier (ooa_reward) scores every candidate against the platform rules and the known-good action sequence
  4. solved -> keep the model's own best answer in self_improved.jsonl (more data in its own successful style)
     never solved -> keep the verified gold flow in hard_fixed.jsonl (learn from mistakes) and queue it for replay
  5. log pass rate, so progress is measured on fresh requests every round
The next SFT run trains from the pretrained base on base data + both files (never fine-tune only on its own output:
that drifts and collapses). Usage:
  python -m scripts.ooa_self_improve --source sft --out DIR --seed 11 --num-prompts 200 --samples 8
"""
import argparse, json, os, sys, time
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="sft")
ap.add_argument("--out", required=True, help="dir holding catalog.jsonl; self-improvement files are written here")
ap.add_argument("--platform", default="/content/platform")
ap.add_argument("--ooa-dir", default="/content/ooa")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--num-prompts", type=int, default=200)
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--max-new-tokens", type=int, default=1024)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=50)
ap.add_argument("--replay", type=int, default=50, help="max earlier failures to retry this round")
args = ap.parse_args()

sys.path.insert(0, args.ooa_dir)
os.environ["OOA_CATALOG"] = os.path.join(args.out, "catalog.jsonl")
os.environ["OOA_SCHEMA"] = os.path.join(args.platform, "contracts/flow-schema/flow.v1.json")
import ooa_reward as R
import gen_flows as G

catalog = [json.loads(l) for l in open(os.environ["OOA_CATALOG"], encoding="utf-8")]


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


# never reuse a flow the model already trained on or was tested on
seen = set()
for f in ("synth_train.jsonl", "self_improved.jsonl", "hard_fixed.jsonl", "eval_synth.jsonl", "rl_val.jsonl"):
    for r in read(os.path.join(args.out, f)):
        if "messages" in r:
            seen.add(r["messages"][1]["content"])
fresh = G.generate(catalog, args.platform, args.num_prompts, args.seed, exclude=seen)
tasks = [{"prompt": q, "gold": mini} for q, _, mini in fresh]
hard_path = os.path.join(args.out, "hard_queue.jsonl")
queue = read(hard_path)[-args.replay:]
tasks += [{"prompt": h["prompt"], "gold": h["gold"]} for h in queue]
print(f"{len(fresh)} fresh requests + {len(queue)} replayed failures")

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval")
engine = Engine(model, tokenizer)
model.eval()

new_good, new_hard, solved_fresh, solved_all, still_hard = [], [], 0, 0, []
rewards = []
parts = {k: 0.0 for k in ("starts_obj", "json", "schema", "actions_exist", "args_ok", "vars_ok", "match")}
n_samples = 0
t0 = time.time()
for ti, t in enumerate(tasks):
    conv = {"messages": [{"role": "user", "content": t["prompt"]}, {"role": "assistant", "content": t["gold"]}]}
    tokens = tokenizer.render_for_completion(conv)
    gold = R.gold_actions(t["gold"])
    seqs, _ = engine.generate_batch(tokens, num_samples=args.samples, max_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_k=args.top_k, seed=args.seed * 1000 + ti)
    scored = []
    for s in seqs:
        text = R.clean(tokenizer.decode(s[len(tokens):]))
        scored.append((R.score(text, gold), text))
    for sc, _ in scored:
        n_samples += 1
        for k in parts:
            parts[k] += sc[k]
    best = max(scored, key=lambda x: x[0]["total"])
    rewards.append(sum(x[0]["total"] for x in scored) / len(scored))
    is_fresh = ti < len(fresh)
    if best[0]["solved"]:
        solved_all += 1
        solved_fresh += is_fresh
        new_good.append({"messages": [{"role": "user", "content": t["prompt"]},
                                      {"role": "assistant", "content": best[1]}]})
    else:
        new_hard.append({"messages": conv["messages"]})
        still_hard.append({"prompt": t["prompt"], "gold": t["gold"]})
    if (ti + 1) % 25 == 0:
        print(f"  {ti + 1}/{len(tasks)} | solved so far {solved_all} | {time.time() - t0:.0f}s")


def append_unique(name, rows):
    path = os.path.join(args.out, name)
    have = {r["messages"][1]["content"] for r in read(path)}
    add = [r for r in rows if r["messages"][1]["content"] not in have]
    with open(path, "a", encoding="utf-8") as f:
        for r in add:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(add)

a = append_unique("self_improved.jsonl", new_good)
b = append_unique("hard_fixed.jsonl", new_hard)
with open(hard_path, "w", encoding="utf-8") as f:   # failures still unsolved are retried next round
    for h in still_hard:
        f.write(json.dumps(h, ensure_ascii=False) + "\n")
stats = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed, "source": args.source,
         "fresh_requests": len(fresh), "fresh_pass_at_k": round(solved_fresh / max(1, len(fresh)), 3),
         "samples": args.samples, "mean_reward": round(sum(rewards) / max(1, len(rewards)), 3),
         "parts": {k: round(v / max(1, n_samples), 3) for k, v in parts.items()},
         "added_self_improved": a, "added_hard_fixed": b, "hard_queue": len(still_hard)}
with open(os.path.join(args.out, "selfimprove_log.jsonl"), "a") as f:
    f.write(json.dumps(stats) + "\n")
print("ROUND RESULT:", json.dumps(stats))
