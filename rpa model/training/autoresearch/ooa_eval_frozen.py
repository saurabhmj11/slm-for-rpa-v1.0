"""Frozen scoreboard for OOA experiments (copy to nanochat/scripts/). NOT to be edited by experiments.
Checks the frozen set's hash, samples EVAL_SAMPLES answers per request from the freshly trained SFT model, and scores each
with the verifier against the FROZEN catalog snapshot. score = mean reward on the generated requests (smooth signal);
real_score = mean reward on the real held-out flows (reality check: long, hard); solved_rate / valid_rate over generated ones."""
import argparse, hashlib, json, os, sys
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

ap = argparse.ArgumentParser()
ap.add_argument("--frozen-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--platform", required=True); ap.add_argument("--ooa-dir", required=True)
args = ap.parse_args()
sys.path.insert(0, args.ooa_dir); sys.path.insert(0, os.path.join(args.ooa_dir, "autoresearch"))
from constants import EVAL_SAMPLES, EVAL_TEMPERATURE, EVAL_MAX_NEW_TOKENS

body = open(os.path.join(args.frozen_dir, "frozen_eval.jsonl"), encoding="utf-8").read()
want = open(os.path.join(args.frozen_dir, "frozen_eval.sha256")).read().strip()
assert hashlib.sha256(body.encode()).hexdigest() == want, "frozen eval was modified: results would not be comparable"
os.environ["OOA_CATALOG"] = os.path.join(args.frozen_dir, "frozen_catalog.jsonl")
os.environ["OOA_SCHEMA"] = os.path.join(args.platform, "contracts/flow-schema/flow.v1.json")
import ooa_reward as R

rows = [json.loads(l) for l in body.splitlines() if l.strip()]
_, _, _, _, device = compute_init(autodetect_device_type())
model, tok, _ = load_model("sft", device, phase="eval")
engine = Engine(model, tok); model.eval()
agg = {"synthetic": [], "real": []}
for i, r in enumerate(rows):
    conv = {"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["gold_flow"]}]}
    tokens = tok.render_for_completion(conv)
    seqs, _ = engine.generate_batch(tokens, num_samples=EVAL_SAMPLES, max_tokens=EVAL_MAX_NEW_TOKENS,
                                    temperature=EVAL_TEMPERATURE, top_k=50, seed=4242 + i)   # fixed seeds: same questions, same dice
    for s in seqs:
        sc = R.score(R.clean(tok.decode(s[len(tokens):])), r["gold_actions"])
        agg[r["kind"]].append(sc)
mean = lambda xs, k: round(sum(x[k] for x in xs) / max(1, len(xs)), 4)
res = {"score": mean(agg["synthetic"], "total"), "real_score": mean(agg["real"], "total"),
       "solved_rate": round(sum(x["solved"] for x in agg["synthetic"]) / max(1, len(agg["synthetic"])), 4),
       "valid_rate": round(sum(x["valid"] for x in agg["synthetic"]) / max(1, len(agg["synthetic"])), 4),
       "parts": {k: mean(agg["synthetic"], k) for k in R.W}}
json.dump(res, open(args.out, "w"))
print(json.dumps(res))
