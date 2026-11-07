"""Generate ooa_nanochat_colab.ipynb: one Colab notebook, nanochat pretrain -> OOA SFT -> eval.
Run:  python make_colab_notebook.py   (writes next to this file)
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
files = {
    "build_catalog.py": (H.parent / "data/catalog/build_catalog.py").read_text(encoding="utf-8"),
    "build_sft_data.py": (H / "build_sft_data.py").read_text(encoding="utf-8"),
    "ooa_task.py": (H / "ooa_task.py").read_text(encoding="utf-8"),
    "gen_flows.py": (H / "gen_flows.py").read_text(encoding="utf-8"),
    "ooa_reward.py": (H / "ooa_reward.py").read_text(encoding="utf-8"),
    "ooa_rl_task.py": (H / "ooa_rl_task.py").read_text(encoding="utf-8"),
    "ooa_self_improve.py": (H / "ooa_self_improve.py").read_text(encoding="utf-8"),
    "ingest_feedback.py": (H / "ingest_feedback.py").read_text(encoding="utf-8"),
    "flow_check.py": (H.parent / "eval/flow_check.py").read_text(encoding="utf-8"),
}

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})

md("""
# OOA model with nanochat: pretrain, then train on our use case

Runs on a free Colab GPU (T4). Order: **Cell 1 config -> Cell 2 setup -> Cells 3-5 pretrain -> Cells 6-8 OOA training and test**.
Everything is saved to your Google Drive, so if Colab disconnects, reconnect and **run all cells again**: finished steps are
skipped and pretraining resumes from the last checkpoint.

Set Runtime -> Change runtime type -> GPU first. Start with `PROFILE = "smoke"` (about 15 minutes) to check the whole pipeline,
then switch to `"t4"` for the real run.

Honest expectations: a free T4 pretrains a small model (tens of millions of parameters, a few hundred million tokens).
It will learn English structure and JSON well enough to be fine-tuned, but it is not a strong general model.
This notebook was written without a GPU to test on, so if a cell errors, send me the error text.
""")

code('''
#@title 1. Config
PROFILE = "smoke"          # "smoke" = quick pipeline check, "t4" = real free-Colab run, "a100" = Colab Pro (A100/L4)
DRIVE_DIR = "/content/drive/MyDrive/ooa-nanochat"   # checkpoints, data and tokenizer live here
NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"   # version this notebook was written against
PLATFORM_REPO = "https://github.com/OneOfficeautomation-Private-Limited/OneOfficeAutomation-Platform"
VOCAB_SIZE = 16384         # tokenizer size (nanochat default is 32768; smaller suits a small model)

PROFILES = {
  #           depth, seq, device_batch, total_batch_tokens, target_tokens, data_shards, tok_chars
  "smoke": dict(depth=4,  seq=512,  dbs=16, total_batch=16384,  tokens=5_000_000,    shards=2, tok_chars=100_000_000),
  "t4":    dict(depth=8,  seq=1024, dbs=8,  total_batch=65536,  tokens=300_000_000,  shards=8, tok_chars=500_000_000),
  "a100":  dict(depth=12, seq=2048, dbs=16, total_batch=262144, tokens=1_000_000_000,shards=16,tok_chars=1_000_000_000),
}
P = PROFILES[PROFILE]
NUM_ITERS = P["tokens"] // P["total_batch"]

# SFT (training on our use case)
SFT_SEQ = 2048             # flows longer than this get cut; the T4 cannot afford much more
SFT_DEVICE_BATCH = 4
SFT_EPOCHS = 2 if PROFILE == "smoke" else 6      # passes over the OOA data (the seed set is small)
FLOW_REPEAT = 15           # each flow example is repeated this many times per pass so flows are about a third of the data
SYNTH_TRAIN = 1500 if PROFILE == "smoke" else 30000   # generated flows (no LLM), see gen_flows.py
SYNTH_EVAL = 20            # unseen generated flows scored by exact action sequence
# Reinforcement layer (after SFT)
RL_PROMPTS = 60 if PROFILE == "smoke" else 400     # requests for RL; each gets RL_SAMPLES answers scored by the verifier
RL_SAMPLES = 8
RL_EXAMPLES_PER_STEP = 8
RL_EVERY = 20 if PROFILE == "smoke" else 25        # evaluate and save every N RL steps
SELF_IMPROVE_ROUNDS = 2
SI_PROMPTS = 20 if PROFILE == "smoke" else 100     # fresh requests per self-improvement round
SI_SAMPLES = 8
EVAL_SOURCE = "sft"        # which model the test cell loads: "sft" or "rl"
SFT_ITERS = 400 if PROFILE == "smoke" else 3000            # upper limit; training also stops when the data runs out
SMOLTALK_ROWS = 3000       # a little general chat data mixed in so the model still talks; 0 = OOA data only
MAX_FLOW_CHARS = 9000      # skip training flows longer than this (about SFT_SEQ tokens)
print(PROFILE, P, "iterations:", NUM_ITERS)
''')

code('''
#@title 2. Setup: GPU, Drive, nanochat, dependencies
import os, subprocess, sys, torch
assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> GPU"
cap = torch.cuda.get_device_capability()
print(torch.cuda.get_device_name(0), "compute capability", cap)
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)
os.environ["NANOCHAT_BASE_DIR"] = DRIVE_DIR + "/nanochat_base"      # checkpoints, tokenizer, data, all persistent
# T4 has no bf16: use fp16 with loss scaling (supported by nanochat). If you see NaN losses, set "float32" (slower).
os.environ["NANOCHAT_DTYPE"] = "bfloat16" if cap >= (8, 0) else "float16"
os.environ["OMP_NUM_THREADS"] = "1"
print("compute dtype:", os.environ["NANOCHAT_DTYPE"])

if not os.path.exists("/content/nanochat"):
    !git clone -q https://github.com/karpathy/nanochat /content/nanochat
!cd /content/nanochat && git checkout -q {NANOCHAT_COMMIT}
# Keep Colab's own torch; install only nanochat's other dependencies
!pip -q install rustbpe tiktoken kernels filelock pyarrow psutil requests wandb jsonschema
%cd /content/nanochat
''')

md("## Pretraining (general language, on nanochat's web-text dataset)")

code('''
#@title 3. Download pretraining data and train the tokenizer (skipped if already done)
tok_pkl = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "tokenizer", "tokenizer.pkl")
!python -m nanochat.dataset -n {P["shards"]}
if not os.path.exists(tok_pkl):
    !python -m scripts.tok_train --max-chars={P["tok_chars"]} --vocab-size={VOCAB_SIZE}
    !python -m scripts.tok_eval
else:
    print("tokenizer already trained:", tok_pkl)
''')

code('''
#@title 4. Pretrain (resumes automatically after a disconnect)
import glob, re
ckpt_dir = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "base_checkpoints", f"d{P['depth']}")
steps = sorted(int(re.search(r"model_(\\d+)\\.pt", f).group(1)) for f in glob.glob(ckpt_dir + "/model_*.pt"))
resume = f"--resume-from-step={steps[-1]}" if steps else ""
if steps and steps[-1] >= NUM_ITERS:
    print("Pretraining already finished at step", steps[-1])
else:
    print("resuming from", steps[-1] if steps else "scratch")
    !python -m scripts.base_train --depth={P["depth"]} --head-dim=64 --window-pattern=L \\
        --max-seq-len={P["seq"]} --device-batch-size={P["dbs"]} --total-batch-size={P["total_batch"]} \\
        --num-iterations={NUM_ITERS} --eval-every=500 --eval-tokens={P["dbs"]*P["seq"]*20} \\
        --core-metric-every=-1 --sample-every=1000 --save-every=250 --run=dummy {resume}
''')

code('''
#@title 5. Quick look at the pretrained model (it continues text; it does not follow instructions yet)
!python -m scripts.base_eval --eval=bpb,sample --device-batch-size=1 --split-tokens=65536 2>&1 | tail -25
''')

md("""
## Training on our use case (OOA flows)

Builds training data from the platform repo with no LLM: requests to Flow JSON from the gold flows (3 wordings each),
plus catalog questions ("which action does X", "what inputs does Y take"). Some workflow families are **held out** and
never trained on; they are used in the final test. This seed set is small (about 60 flow examples): replace
`train.jsonl` with the full dialogue set when you have it.
""")

code('''
#@title 6. Build OOA training data
import json
os.makedirs("/content/ooa", exist_ok=True)
FILES = %s
for name, text in FILES.items():
    open(f"/content/ooa/{name}", "w", encoding="utf-8").write(text)
if not os.path.exists("/content/platform"):
    !git clone -q --depth 1 {PLATFORM_REPO} /content/platform
!python /content/ooa/build_sft_data.py /content/platform {DRIVE_DIR}/sft {MAX_FLOW_CHARS}
!python /content/ooa/gen_flows.py {DRIVE_DIR}/sft/catalog.jsonl /content/platform {DRIVE_DIR}/sft/synth_train.jsonl {SYNTH_TRAIN} 1
!python /content/ooa/gen_flows.py {DRIVE_DIR}/sft/catalog.jsonl /content/platform /content/synth_eval_raw.jsonl 400 2
train_set = {json.loads(l)["messages"][1]["content"] for l in open(f"{DRIVE_DIR}/sft/synth_train.jsonl")}
ev = []
for l in open("/content/synth_eval_raw.jsonl"):
    m = json.loads(l)["messages"]
    if m[1]["content"] not in train_set and len(ev) < SYNTH_EVAL:
        ev.append({"prompt": m[0]["content"], "gold_actions": [st["action"] for st in json.loads(m[1]["content"])["steps"]]})
open(f"{DRIVE_DIR}/sft/eval_synth.jsonl", "w").write("\\n".join(json.dumps(x) for x in ev))
# validation set the model never trains on (the old one overlapped training, so its loss looked great for nothing)
val_rows = [json.loads(l) for l in open(f"{DRIVE_DIR}/sft/val.jsonl")]
val_rows = [r for r in val_rows if r["messages"][1]["content"].startswith("Use ") or r["messages"][1]["content"].startswith("`")]
raw = [json.loads(l) for l in open("/content/synth_eval_raw.jsonl")]
extra = [r for r in raw[SYNTH_EVAL:] if r["messages"][1]["content"] not in train_set][:150]
with open(f"{DRIVE_DIR}/sft/val.jsonl", "w") as f:
    for r in val_rows + extra:
        f.write(json.dumps(r) + "\\n")
print("validation rows (unseen):", len(val_rows) + len(extra))
print("synthetic eval prompts (not in train):", len(ev))
!cp /content/ooa/ooa_task.py /content/nanochat/tasks/ooa.py
!head -c 700 {DRIVE_DIR}/sft/train.jsonl
''' % json.dumps(files))

code('''
#@title 7. Patch nanochat SFT to use our data, then train
import re, shutil
# always start from the pinned nanochat scripts, so an earlier patch can never hide the current data mix
!cd /content/nanochat && git checkout -q -- scripts/chat_sft.py scripts/chat_cli.py
p = "/content/nanochat/scripts/chat_sft.py"
s = open(p).read()
if "OOADialogues" not in s:
    s = s.replace("from tasks.smoltalk import SmolTalk", "from tasks.smoltalk import SmolTalk\\nfrom tasks.ooa import OOADialogues")
    a = s.index("train_tasks = [")
    b = s.index("\\n", s.index("# total: 24K"))
    new = f"""train_tasks = ([OOADialogues("{DRIVE_DIR}/sft/train_flows.jsonl") for _ in range({FLOW_REPEAT})]
               + [OOADialogues("{DRIVE_DIR}/sft/train_catalog.jsonl"), OOADialogues("{DRIVE_DIR}/sft/synth_train.jsonl")]) * {SFT_EPOCHS}
import os as _os
_extra = [OOADialogues("{DRIVE_DIR}/sft/" + _f) for _f in ("self_improved.jsonl", "hard_fixed.jsonl", "feedback_train.jsonl")
          if _os.path.exists("{DRIVE_DIR}/sft/" + _f)]
train_tasks += _extra * {SFT_EPOCHS}
print0(f"extra self-learning files in the mix: {{len(_extra)}}")
if {SMOLTALK_ROWS} > 0:
    train_tasks.append(SmolTalk(split="train", stop={SMOLTALK_ROWS}))
train_dataset = TaskMixture(train_tasks)
print0(f"OOA training mixture: {{len(train_dataset):,}} rows")
val_dataset = TaskMixture([OOADialogues("{DRIVE_DIR}/sft/val.jsonl")])"""
    s = s[:a] + new + s[b:]
    open(p, "w").write(s)
    print("patched chat_sft.py")
# remove old SFT checkpoints: nanochat loads the highest step number, so a stale step-100 would hide a newer, shorter run
import shutil
shutil.rmtree(os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatsft_checkpoints"), ignore_errors=True)
!python -m scripts.chat_sft --max-seq-len={SFT_SEQ} --device-batch-size={SFT_DEVICE_BATCH} \\
    --total-batch-size={SFT_SEQ*SFT_DEVICE_BATCH*4} --num-iterations={SFT_ITERS} \\
    --eval-every=100 --eval-tokens={SFT_SEQ*SFT_DEVICE_BATCH*8} --chatcore-every=-1 --run=dummy
''')

code('''
#@title 8. Test on held-out workflow families (never trained on)
import subprocess, json, sys
EVAL_SOURCE = globals().get("EVAL_SOURCE", "sft")
print("evaluating model source:", EVAL_SOURCE)
# chat_cli stops after 256 new tokens; flows are longer, so raise the limit
cli = "/content/nanochat/scripts/chat_cli.py"
t = open(cli).read()
t = t.replace('"max_tokens": 256', '"max_tokens": 3000')
open(cli, "w").write(t)
sys.path.insert(0, "/content/ooa")
import jsonschema
schema = json.load(open("/content/platform/contracts/flow-schema/flow.v1.json"))
catalog = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DRIVE_DIR}/sft/catalog.jsonl")}

def steps(nodes):
    for s in nodes:
        yield s
        for k in ("children", "else"):
            if isinstance(s.get(k), list): yield from steps(s[k])
        for ei in s.get("elseIf") or []: yield from steps(ei.get("children", []))
        for c in s.get("catch") or []: yield from steps(c.get("children", []))
        if isinstance(s.get("finally"), list): yield from steps(s["finally"])

results = []
for row in map(json.loads, open(f"{DRIVE_DIR}/sft/eval.jsonl")):
    out = subprocess.run(["python", "-m", "scripts.chat_cli", "-i", EVAL_SOURCE, "-p", row["prompt"], "-t", "0.2"],
                         capture_output=True, text=True, cwd="/content/nanochat").stdout
    text = out.split("Assistant:", 1)[-1].split("<|assistant_end|>")[0].strip()   # drop the end-of-answer marker
    print("---", row["family"], "| first 250 chars:", text[:250])
    r = {"family": row["family"], "json": False, "schema": False, "actions_known": False}
    try:
        flow = json.loads(text); r["json"] = True
        jsonschema.validate(flow, schema); r["schema"] = True
        acts = [s.get("action") for s in steps(flow.get("steps", [])) if isinstance(s.get("action"), str) and "/" in s["action"]]
        r["actions_known"] = all(a in catalog for a in acts)
    except Exception as e:
        r["error"] = str(e)[:80]
    results.append(r); print(r)
n = len(results)
print(f"\\nheld-out flows: {n} | valid JSON {sum(r['json'] for r in results)} | schema OK {sum(r['schema'] for r in results)} | all actions exist {sum(r['actions_known'] for r in results)}")
print("\\n=== Synthetic eval: unseen generated flows, exact action sequence ===")
sr = []
for row in map(json.loads, open(f"{DRIVE_DIR}/sft/eval_synth.jsonl")):
    out = subprocess.run(["python", "-m", "scripts.chat_cli", "-i", EVAL_SOURCE, "-p", row["prompt"], "-t", "0.2"],
                         capture_output=True, text=True, cwd="/content/nanochat").stdout
    text = out.split("Assistant:", 1)[-1].split("<|assistant_end|>")[0].strip()
    r = {"json": False, "schema": False, "actions_known": False, "actions_match": False}
    try:
        flow = json.loads(text); r["json"] = True
        jsonschema.validate(flow, schema); r["schema"] = True
        acts = [x.get("action") for x in steps(flow.get("steps", []))]
        r["actions_known"] = all(a in catalog for a in acts)
        r["actions_match"] = acts == row["gold_actions"]
    except Exception:
        pass
    sr.append(r)
print(f"synthetic eval: {len(sr)} | valid JSON {sum(r['json'] for r in sr)} | schema OK {sum(r['schema'] for r in sr)} | actions exist {sum(r['actions_known'] for r in sr)} | exact action sequence {sum(r['actions_match'] for r in sr)}")
print("\\nTry it yourself:")
!cd /content/nanochat && python -m scripts.chat_cli -i sft -p "Which OOA action would I use to read a CSV file into a table?" -t 0.2
''')

import colab_rl_cells
colab_rl_cells.build(md, code)

md("""
## After this run
- Checkpoints: `DRIVE_DIR/nanochat_base/base_checkpoints` (pretrained) and `chatsft_checkpoints` (OOA-trained).
- Weak results on the held-out test are expected with the seed data. The next lever is more and better dialogues
  (edits, error fixing, missing-input questions), validated with the scorer in `eval/flow_check.py`.
- Reinforcement layer: cells 9-10 (RL with the verifier as reward; self-improvement rounds). Feed real usage back with `ingest_feedback.py`.
- For a stronger base, use `PROFILE = "t4"` and let it run (resume after disconnects), or an A100 on Colab Pro.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 0}
out = H / "ooa_nanochat_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
