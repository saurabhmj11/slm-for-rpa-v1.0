"""Notebook cells for the reinforcement layer (imported by make_colab_notebook.py)."""


def build(md, code):
    md("""
## Reinforcement layer: the model learns from the platform's own rules

Two loops, both scored by a **verifier** (`ooa_reward.py`: schema, actions exist, required inputs, declared variables,
action sequence vs the known-good flow). No human labels and no LLM judge are needed.

1. **RL (cell 9):** nanochat's `chat_rl.py` writes 8 answers per request; answers the verifier scores above the group
   average are reinforced. Partial credit gives a gradient even when no answer is perfect.
2. **Self-improvement rounds (cell 10):** each round the model attempts *fresh* requests. Verified successes are kept as new
   training data; requests it fails on are kept with the verified correct flow and retried next round. Each round then
   retrains from the pretrained base on all data so far (never only on its own output, which drifts). The pass rate on
   fresh requests per round is the progress measure.

Run cell 7 (SFT) once before these. Real user feedback joins later through `ingest_feedback.py`.
""")

    code('''
#@title 9. Reinforcement learning with the verifier as reward
import json, os, shutil, sys
os.environ["OOA_CATALOG"] = f"{DRIVE_DIR}/sft/catalog.jsonl"
os.environ["OOA_SCHEMA"] = "/content/platform/contracts/flow-schema/flow.v1.json"
sys.path.insert(0, "/content/ooa")
import gen_flows as G
catalog = [json.loads(l) for l in open(os.environ["OOA_CATALOG"], encoding="utf-8")]
used = set()
for f in ("synth_train.jsonl", "self_improved.jsonl", "hard_fixed.jsonl"):
    if os.path.exists(f"{DRIVE_DIR}/sft/{f}"):
        used |= {json.loads(l)["messages"][1]["content"] for l in open(f"{DRIVE_DIR}/sft/{f}")}
def rows(n, seed, exclude):
    out = G.generate(catalog, "/content/platform", n, seed, exclude=exclude)
    out.sort(key=lambda t: len(t[1]["steps"]))      # short (easy) requests first: a simple curriculum
    return out
tr = rows(RL_PROMPTS, 3, used)
va = rows(40, 4, used | {m for _, _, m in tr})
for name, data in (("rl_train", tr), ("rl_val", va)):
    with open(f"{DRIVE_DIR}/sft/{name}.jsonl", "w", encoding="utf-8") as f:
        for q, _, mini in data:
            f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": mini}]}) + "\\n")
print("RL requests:", len(tr), "train /", len(va), "val")

!cp /content/ooa/ooa_rl_task.py /content/nanochat/tasks/ooa_rl.py
!cd /content/nanochat && git checkout -q -- scripts/chat_rl.py
p = "/content/nanochat/scripts/chat_rl.py"
s = open(p).read()
for old, new in [
    ("from tasks.gsm8k import GSM8K", "from tasks.ooa_rl import OOAFlowRL"),
    ('train_task = GSM8K(subset="main", split="train")', f'train_task = OOAFlowRL("{DRIVE_DIR}/sft/rl_train.jsonl")'),
    ('val_task = GSM8K(subset="main", split="test")', f'val_task = OOAFlowRL("{DRIVE_DIR}/sft/rl_val.jsonl")'),
    ("max_completion_tokens=256,", "max_completion_tokens=1024,"),
]:
    assert old in s, "chat_rl.py changed upstream, cannot patch: " + old
    s = s.replace(old, new)
open(p, "w").write(s)
shutil.rmtree(os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatrl_checkpoints"), ignore_errors=True)
!cd /content/nanochat && python -m scripts.chat_rl --num-samples={RL_SAMPLES} --device-batch-size={RL_SAMPLES} \\
    --examples-per-step={RL_EXAMPLES_PER_STEP} --max-new-tokens=1024 --eval-every={RL_EVERY} --eval-examples=40 \\
    --save-every={RL_EVERY} --num-epochs=1 --run=dummy
''')

    code('''
#@title 9b. Point the test at the RL model, then re-run cell 8
EVAL_SOURCE = "rl"     # cell 8 prints "evaluating model source: rl". Set back to "sft" to test the SFT model.
''')

    code('''
#@title 10. Self-improvement rounds: attempt fresh requests, keep verified wins, learn from failures, retrain
import json, os, shutil
os.environ["OOA_CATALOG"] = f"{DRIVE_DIR}/sft/catalog.jsonl"
os.environ["OOA_SCHEMA"] = "/content/platform/contracts/flow-schema/flow.v1.json"
!cp /content/ooa/ooa_self_improve.py /content/nanochat/scripts/ooa_self_improve.py
# guards: cell 10 reuses the data and the patched chat_sft.py from cells 6-7. Stop if they do not match the config.
import re as _re
_n = sum(1 for _ in open(f"{DRIVE_DIR}/sft/synth_train.jsonl"))
_patched = open("/content/nanochat/scripts/chat_sft.py").read()
_ep = _re.search(r"\\) \\* (\\d+)\\n", _patched)
print(f"synth_train.jsonl rows: {_n} (config SYNTH_TRAIN={SYNTH_TRAIN}) | patched SFT epochs: {_ep.group(1) if _ep else '?'} (config {SFT_EPOCHS})")
if _n < 0.9 * SYNTH_TRAIN or (_ep and int(_ep.group(1)) != SFT_EPOCHS):
    raise RuntimeError("Data or SFT patch is out of date for the current config. Run cell 1, cell 6 (data), cell 7 (SFT), then run this cell again.")
sft_cmd = (f"python -m scripts.chat_sft --max-seq-len={SFT_SEQ} --device-batch-size={SFT_DEVICE_BATCH} "
           f"--total-batch-size={SFT_SEQ*SFT_DEVICE_BATCH*4} --num-iterations={SFT_ITERS} "
           f"--eval-every=100 --eval-tokens={SFT_SEQ*SFT_DEVICE_BATCH*8} --chatcore-every=-1 --run=dummy")
for r in range(SELF_IMPROVE_ROUNDS):
    print(f"\\n===== self-improvement round {r + 1}/{SELF_IMPROVE_ROUNDS} =====")
    !cd /content/nanochat && python -m scripts.ooa_self_improve --source sft --out {DRIVE_DIR}/sft --seed {100 + r} --num-prompts {SI_PROMPTS} --samples {SI_SAMPLES}
    _ck = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatsft_checkpoints")
    if os.path.exists(_ck):                      # keep the previous model (never deleted), like cell 7
        import time as _t
        shutil.move(_ck, _ck + "_before_" + _t.strftime("%Y%m%d_%H%M%S"))
    !cd /content/nanochat && {sft_cmd}
print("\\nPass rate on FRESH requests, per round (this is the progress measure):")
for l in open(f"{DRIVE_DIR}/sft/selfimprove_log.jsonl"):
    x = json.loads(l)
    print(x["time"], "| pass@k", x["fresh_pass_at_k"], "| mean reward", x["mean_reward"], "| new wins", x["added_self_improved"], "| new fixes", x["added_hard_fixed"], "| still hard", x["hard_queue"])
    print("   where answers lose points (1.0 = always fine):", x.get("parts"))
''')
