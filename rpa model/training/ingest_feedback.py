"""Turn real usage into training data (the long-run self-learning signal).

Studio (or any client) appends one JSON line per suggestion to feedback.jsonl:
  {"prompt": "...", "flow": {...flow json...}, "accepted": true|false, "edited_flow": {...optional...}}
Rules (humans are a signal, the verifier is the gate):
  accepted, or edited_flow given -> candidate training example (edited_flow wins: it is the human's correction)
  every candidate must pass the verifier (schema, actions exist, args, variables) or it is dropped
  rejected with no edit -> not trained on; listed in rejected_prompts.jsonl as hard cases to write good answers for
Usage: python ingest_feedback.py FEEDBACK.jsonl OUT_DIR CATALOG.jsonl FLOW_SCHEMA.json
Writes OUT_DIR/feedback_train.jsonl (nanochat format) and OUT_DIR/rejected_prompts.jsonl
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
fb, out, cat, schema = sys.argv[1:5]
os.environ["OOA_CATALOG"], os.environ["OOA_SCHEMA"] = cat, schema
import ooa_reward as R

good, rejected, dropped = [], [], 0
for l in open(fb, encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    flow = r.get("edited_flow") or (r.get("flow") if r.get("accepted") else None)
    if flow is None:
        rejected.append({"prompt": r["prompt"]})
        continue
    mini = json.dumps(flow, separators=(",", ":"), ensure_ascii=False)
    if not R.score(mini)["valid"]:
        dropped += 1
        continue
    good.append({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": mini}]})
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "feedback_train.jsonl"), "w", encoding="utf-8") as f:
    for g in good:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")
with open(os.path.join(out, "rejected_prompts.jsonl"), "w", encoding="utf-8") as f:
    for g in rejected:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")
print(f"{len(good)} usable, {dropped} dropped (failed verifier), {len(rejected)} rejected-without-edit")
