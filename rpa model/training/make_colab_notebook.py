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
    "autoresearch/constants.py": (H / "autoresearch/constants.py").read_text(encoding="utf-8"),
    "autoresearch/recipe.py": (H / "autoresearch/recipe.py").read_text(encoding="utf-8"),
    "autoresearch/loop.py": (H / "autoresearch/loop.py").read_text(encoding="utf-8"),
    "autoresearch/run_experiment.py": (H / "autoresearch/run_experiment.py").read_text(encoding="utf-8"),
    "autoresearch/ooa_eval_frozen.py": (H / "autoresearch/ooa_eval_frozen.py").read_text(encoding="utf-8"),
    "retrieval.py": (H / "retrieval.py").read_text(encoding="utf-8"),
    "repair.py": (H / "repair.py").read_text(encoding="utf-8"),
    "data_filter.py": (H / "data_filter.py").read_text(encoding="utf-8"),
    "ooa_format.py": (H / "ooa_format.py").read_text(encoding="utf-8"),
    "render_dataset.py": (H / "render_dataset.py").read_text(encoding="utf-8"),
    "safety.py": (H / "safety.py").read_text(encoding="utf-8"),
    "model_backend.py": (H.parent / "assistant/model_backend.py").read_text(encoding="utf-8"),
    "constrained.py": (H.parent / "assistant/constrained.py").read_text(encoding="utf-8"),
    "serving/ooa_server.py": (H.parent / "serving/ooa_server.py").read_text(encoding="utf-8"),
    "serving/backends.py": (H.parent / "serving/backends.py").read_text(encoding="utf-8"),
    "flow_check.py": (H.parent / "eval/flow_check.py").read_text(encoding="utf-8"),
    "plain_requests.jsonl": (H.parent / "eval/plain_requests.jsonl").read_text(encoding="utf-8"),
    "plain_requests_2.jsonl": (H.parent / "eval/plain_requests_2.jsonl").read_text(encoding="utf-8"),
    "plain_requests_3.jsonl": (H.parent / "eval/plain_requests_3.jsonl").read_text(encoding="utf-8"),
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
PROFILE = "t4"             # "t4" = real free-Colab run, "smoke" = quick pipeline check (tiny model), "a100" = Colab Pro (A100/L4)
DRIVE_DIR = "/content/drive/MyDrive/ooa-nanochat"   # checkpoints, data and tokenizer live here
SHARED_FOLDER_ID = ""      # optional: Drive ID of the ooa-nanochat folder shared from another Google account (from its link)
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
NATURAL_REPEAT = 5         # passes over sft/natural_train.jsonl per epoch, if that file is in Drive (everyday wording)
CORE_REPEAT = 5            # passes over sft/core_train.jsonl per epoch, if in Drive (everyday requests for the 65 most used actions)
REBUILD_DATA = False       # cell 6 reuses the generated flows already in Drive (saves time); True = generate them again
FLOW_REPEAT = 5            # real flows repeated per pass; was 15, which made the model recite whole training flows (memorising)
SYNTH_TRAIN = 1500 if PROFILE == "smoke" else 30000   # generated flows (no LLM), see gen_flows.py
SYNTH_EVAL = 20            # unseen generated flows scored by exact action sequence
SYNTH_PATTERN = 2000 if PROFILE == "smoke" else 12000   # long, multi-toolkit flows following real-flow patterns (gen_flows.real_patterns)
# Reinforcement layer (after SFT)
RL_PROMPTS = 60 if PROFILE == "smoke" else 400     # requests for RL; each gets RL_SAMPLES answers scored by the verifier
RL_SAMPLES = 8
RL_EXAMPLES_PER_STEP = 8
RL_EVERY = 20 if PROFILE == "smoke" else 25        # evaluate and save every N RL steps
SELF_IMPROVE_ROUNDS = 2
SI_PROMPTS = 20 if PROFILE == "smoke" else 100     # fresh requests per self-improvement round
SI_SAMPLES = 8
EVAL_SOURCE = "sft"        # which model the test cell loads: "sft" or "rl"
SFT_ITERS = 400 if PROFILE == "smoke" else 16000           # counted in MICRO-batches (4 per optimizer step): 16000 = 4000 steps, about 75 min on a T4 (was 10000: too short for long flows)
CLEAN_DATA = True          # cell 6 runs the strict data filter (sft/ -> sft_clean/) and cell 7 trains on the cleaned data
SCEN_REPEAT = 15           # whole business processes are long and few: weight them 3x the other files (CORE_REPEAT)
RENDER = True              # cell 6d writes the data in the plan-first format with retrieval context (sft_render/); cell 7 trains on it
NEW_FILES = {"long_train.jsonl": 15, "business2_train.jsonl": 15, "targeted_train.jsonl": 8,   # new files in Drive sft/
             "fix_train.jsonl": 5, "clarify_train.jsonl": 5}                                     # -> times per epoch
ASSISTANT_EVAL = True      # cell 8 also scores the full assistant (real action names only, repair, loop guard), as users get it
SMOLTALK_ROWS = 3000       # a little general chat data mixed in so the model still talks; 0 = OOA data only
MAX_FLOW_CHARS = 9000      # skip training flows longer than this (about SFT_SEQ tokens)
print(PROFILE, P, "iterations:", NUM_ITERS)
print("SFT settings -> SYNTH_TRAIN:", SYNTH_TRAIN, "| SFT_EPOCHS:", SFT_EPOCHS, "| SFT_ITERS cap:", SFT_ITERS, "| CLEAN_DATA:", CLEAN_DATA)
''')

code('''
#@title 2. Setup: GPU, Drive, nanochat, dependencies
import os, subprocess, sys, torch, glob
assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> GPU"
cap = torch.cuda.get_device_capability()
print(torch.cuda.get_device_name(0), "compute capability", cap)
from google.colab import drive
drive.mount("/content/drive")
# a folder shared from another Google account is reached by its ID (needs "Add shortcut to My Drive" once in this account)
if SHARED_FOLDER_ID:
    hits = glob.glob(f"/content/drive/.shortcut-targets-by-id/{SHARED_FOLDER_ID}/**/nanochat_base/base_checkpoints", recursive=True)
    if hits:
        DRIVE_DIR = os.path.dirname(os.path.dirname(hits[0]))
    print("shared folder:", DRIVE_DIR if hits else "NOT FOUND, using " + DRIVE_DIR)
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
    os.makedirs(os.path.dirname(f"/content/ooa/{name}"), exist_ok=True)
    open(f"/content/ooa/{name}", "w", encoding="utf-8").write(text)
if not os.path.exists("/content/platform"):
    !git clone -q --depth 1 {PLATFORM_REPO} /content/platform
!python /content/ooa/build_sft_data.py /content/platform {DRIVE_DIR}/sft {MAX_FLOW_CHARS}
def _have(name, rows):        # reuse a generated file from an earlier run unless REBUILD_DATA
    f = f"{DRIVE_DIR}/sft/{name}"
    ok = (not REBUILD_DATA) and os.path.exists(f) and sum(1 for _ in open(f)) >= rows * 0.9
    print(("reusing " if ok else "generating ") + name)
    return ok
if not _have("synth_train.jsonl", SYNTH_TRAIN):
    !python /content/ooa/gen_flows.py {DRIVE_DIR}/sft/catalog.jsonl /content/platform {DRIVE_DIR}/sft/synth_train.jsonl {SYNTH_TRAIN} 1
!python /content/ooa/gen_flows.py {DRIVE_DIR}/sft/catalog.jsonl /content/platform /content/synth_eval_raw.jsonl 400 2
train_set = {json.loads(l)["messages"][1]["content"] for l in open(f"{DRIVE_DIR}/sft/synth_train.jsonl")}
def _all_actions(nodes):   # every step, including those inside if / try / loop (the test compares all of them)
    out = []
    for n in nodes or []:
        out.append(n.get("action"))
        for k in ("children", "else", "finally"):
            if isinstance(n.get(k), list): out += _all_actions(n[k])
        for ei in n.get("elseIf") or []: out += _all_actions(ei.get("children"))
        for c in n.get("catch") or []: out += _all_actions(c.get("children"))
    return out
ev = []
for l in open("/content/synth_eval_raw.jsonl"):
    m = json.loads(l)["messages"]
    if m[1]["content"] not in train_set and len(ev) < SYNTH_EVAL:
        ev.append({"prompt": m[0]["content"], "gold_actions": _all_actions(json.loads(m[1]["content"])["steps"])})
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
if not _have("synth_pattern.jsonl", SYNTH_PATTERN):
    !python /content/ooa/gen_flows.py {DRIVE_DIR}/sft/catalog.jsonl /content/platform {DRIVE_DIR}/sft/synth_pattern.jsonl {SYNTH_PATTERN} 1 patterns
!python /content/ooa/gen_flows.py {DRIVE_DIR}/sft/catalog.jsonl /content/platform /content/pattern_eval_raw.jsonl 200 2 patterns
_ptrain = {json.loads(l)["messages"][1]["content"] for l in open(f"{DRIVE_DIR}/sft/synth_pattern.jsonl")}
_pev = []
for l in open("/content/pattern_eval_raw.jsonl"):
    m = json.loads(l)["messages"]
    if m[1]["content"] not in _ptrain and len(_pev) < SYNTH_EVAL:
        _f = json.loads(m[1]["content"])
        _pev.append({"prompt": m[0]["content"], "gold_flow": m[1]["content"]})
open(f"{DRIVE_DIR}/sft/eval_pattern.jsonl", "w").write("\\n".join(json.dumps(x) for x in _pev))
print("real-pattern eval prompts (not in train):", len(_pev))
!cp /content/ooa/ooa_task.py /content/nanochat/tasks/ooa.py
# the catalog was just rebuilt from the platform downloaded in cell 2, so data and checks match the current platform
print("actions in the catalog:", sum(1 for _ in open(f"{DRIVE_DIR}/sft/catalog.jsonl")))
if CLEAN_DATA:
    # strict filter: structure, length, leaked variable names, repeated clauses, platform rules (after adding missing
    # open/save steps), stale actions, wrong app, closeness to a test request, duplicates, conflicts, over-represented
    # action sequences. Report: sft_clean/filter_report.json
    !python /content/ooa/data_filter.py {DRIVE_DIR}/sft {DRIVE_DIR}/sft_clean /content/platform /content/ooa --cap 60
''' % json.dumps(files))

code('''
#@title 6d. Write the data in the plan-first format (new): sft_clean + the new files -> sft_render
# Every request gets a "### Context" block (the likely actions, 0-2 similar examples, sometimes the agent's variables)
# and every answer becomes "Plan: 1. ... 2. ..." then "Flow: {...}". Flows too long for the 2048-token window are
# written as their parts. Variables are written $name$ as the platform now saves them. About 3-5 minutes.
import glob, shutil
_src = f"{DRIVE_DIR}/sft_clean" if CLEAN_DATA and os.path.exists(f"{DRIVE_DIR}/sft_clean/filter_report.json") else f"{DRIVE_DIR}/sft"
_extra = [f"{DRIVE_DIR}/sft/{f}" for f in NEW_FILES if os.path.exists(f"{DRIVE_DIR}/sft/{f}")]
for f in NEW_FILES:
    if not os.path.exists(f"{DRIVE_DIR}/sft/{f}"):
        print("MISSING in Drive sft/ (upload it):", f)
for f in glob.glob(f"{DRIVE_DIR}/sft/* (*).jsonl"):
    print("WARNING: Drive copy with (n) in the name, delete it and keep one file:", f)
if RENDER:
    if os.path.exists(f"{DRIVE_DIR}/sft_render"):
        shutil.rmtree(f"{DRIVE_DIR}/sft_render")
    _x = " ".join(_extra)
    !cd /content/ooa && python render_dataset.py {_src} {DRIVE_DIR}/sft_render --catalog {DRIVE_DIR}/sft/catalog.jsonl --extra {_x}
    # catalog question rows stay as they are
    shutil.copy(f"{_src}/train_catalog.jsonl", f"{DRIVE_DIR}/sft_render/train_catalog.jsonl")
    _r = json.load(open(f"{DRIVE_DIR}/sft_render/render_report.json"))
    print("rendered from", _src, "+", len(_extra), "new files ->", f"{DRIVE_DIR}/sft_render")
    print(json.dumps({k: v for k, v in _r.items() if not isinstance(v, (dict, list))}, indent=1))
''')

code('''
#@title 7. Patch nanochat SFT to use our data, then train
import re, shutil
SFT_DIR = f"{DRIVE_DIR}/sft_clean" if CLEAN_DATA and os.path.exists(f"{DRIVE_DIR}/sft_clean/filter_report.json") else f"{DRIVE_DIR}/sft"
if RENDER and os.path.exists(f"{DRIVE_DIR}/sft_render/render_report.json"):
    SFT_DIR = f"{DRIVE_DIR}/sft_render"
print("training data:", SFT_DIR)
# always start from the pinned nanochat scripts, so an earlier patch can never hide the current data mix
!cd /content/nanochat && git checkout -q -- scripts/chat_sft.py scripts/chat_cli.py
p = "/content/nanochat/scripts/chat_sft.py"
s = open(p).read()
if "OOADialogues" not in s:
    s = s.replace("from tasks.smoltalk import SmolTalk", "from tasks.smoltalk import SmolTalk\\nfrom tasks.ooa import OOADialogues")
    a = s.index("train_tasks = [")
    b = s.index("\\n", s.index("# total: 24K"))
    new = f"""train_tasks = ([OOADialogues("{SFT_DIR}/train_flows.jsonl") for _ in range({FLOW_REPEAT})]
               + [OOADialogues("{SFT_DIR}/train_catalog.jsonl"), OOADialogues("{SFT_DIR}/synth_train.jsonl"),
                  OOADialogues("{SFT_DIR}/synth_pattern.jsonl")]) * {SFT_EPOCHS}
import os as _os
_extra = [OOADialogues("{SFT_DIR}/" + _f) for _f in ("self_improved.jsonl", "hard_fixed.jsonl", "feedback_train.jsonl")
          if _os.path.exists("{SFT_DIR}/" + _f)]
train_tasks += _extra * {SFT_EPOCHS}
print0(f"extra self-learning files in the mix: {{len(_extra)}}")
_nat = "{SFT_DIR}/natural_train.jsonl"     # everyday-wording requests for generated flows (upload to Drive sft/)
if _os.path.exists(_nat):
    train_tasks += [OOADialogues(_nat)] * ({NATURAL_REPEAT} * {SFT_EPOCHS})
print0(f"natural-wording file in the mix: {{_os.path.exists(_nat)}}")
_core = "{SFT_DIR}/core_train.jsonl"       # everyday requests for the most used actions (upload to Drive sft/)
if _os.path.exists(_core):
    train_tasks += [OOADialogues(_core)] * ({CORE_REPEAT} * {SFT_EPOCHS})
print0(f"core-actions file in the mix: {{_os.path.exists(_core)}}")
_chains = "{SFT_DIR}/chains_train.jsonl"   # everyday multi-step chains across tools (upload to Drive sft/)
if _os.path.exists(_chains):
    train_tasks += [OOADialogues(_chains)] * ({CORE_REPEAT} * {SFT_EPOCHS})
print0(f"chains file in the mix: {{_os.path.exists(_chains)}}")
_combos = "{SFT_DIR}/combos_train.jsonl"   # many different combinations, loops and conditions (upload to Drive sft/)
if _os.path.exists(_combos):
    train_tasks += [OOADialogues(_combos)] * ({CORE_REPEAT} * {SFT_EPOCHS})
print0(f"combinations file in the mix: {{_os.path.exists(_combos)}}")
_notask = "{SFT_DIR}/notask_train.jsonl"   # questions / chat / pasted errors -> short text reply, not a flow
if _os.path.exists(_notask):
    train_tasks += [OOADialogues(_notask)] * ({CORE_REPEAT} * {SFT_EPOCHS})
print0(f"not-a-task file in the mix: {{_os.path.exists(_notask)}}")
_tools = "{SFT_DIR}/tools_train.jsonl"   # business apps (Google, Microsoft 365, Asana, GitHub, Slack ...)
if _os.path.exists(_tools):
    train_tasks += [OOADialogues(_tools)] * ({CORE_REPEAT} * {SFT_EPOCHS})
print0(f"business-apps file in the mix: {{_os.path.exists(_tools)}}")
_scen = "{SFT_DIR}/scenarios_train.jsonl"   # whole business processes (3-8 steps, loops, conditions)
if _os.path.exists(_scen):
    train_tasks += [OOADialogues(_scen)] * ({SCEN_REPEAT} * {SFT_EPOCHS})
print0(f"business-processes file in the mix: {{_os.path.exists(_scen)}}")
for _f, _n in {NEW_FILES}.items():       # the new files: long processes, business areas, targeted, self-correction, questions
    if _os.path.exists("{SFT_DIR}/" + _f):
        train_tasks += [OOADialogues("{SFT_DIR}/" + _f)] * (_n * {SFT_EPOCHS})
    print0(f"{{_f}} in the mix: {{_os.path.exists('{SFT_DIR}/' + _f)}}")
if {SMOLTALK_ROWS} > 0:
    train_tasks.append(SmolTalk(split="train", stop={SMOLTALK_ROWS}))
train_dataset = TaskMixture(train_tasks)
print0(f"OOA training mixture: {{len(train_dataset):,}} rows")
val_dataset = TaskMixture([OOADialogues("{SFT_DIR}/val.jsonl")])"""
    s = s[:a] + new + s[b:]
    # never train on a cut-off flow: nanochat truncates conversations to 2048 tokens; skip ones longer than a row instead
    old_r = "            ids, mask = tokenizer.render_conversation(conversation)\\n            conv_buffer.append((ids, mask))"
    new_r = ("            ids, mask = tokenizer.render_conversation(conversation, max_tokens=10**9)\\n"
             "            if len(ids) <= row_capacity:\\n"
             "                conv_buffer.append((ids, mask))\\n"
             "            else:\\n"
             "                skipped_long[0] += 1")
    assert old_r in s, "chat_sft.py loader changed upstream"
    s = s.replace(old_r, new_r)
    s = s.replace("last_step = False # we will toggle this", "skipped_long = [0]\\nlast_step = False # we will toggle this")
    s = s.replace('print0(f"Total training time:', 'print0(f"Conversations skipped as longer than the window (not cut): {skipped_long[0]}")\\nprint0(f"Total training time:')
    open(p, "w").write(s)
    print("patched chat_sft.py")
# keep the previous model: nanochat loads the highest step in chatsft_checkpoints, so move it aside (never deleted)
import shutil, time
_ck = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatsft_checkpoints")
if os.path.exists(_ck):
    _keep = _ck + "_before_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.move(_ck, _keep)
    print("previous model kept in", _keep)
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
# load the model ONCE (it used to be reloaded for every test request through chat_cli, which was slow)
sys.path.insert(0, "/content/nanochat")
import torch
from nanochat.checkpoint_manager import load_model as _load_model
from nanochat.engine import Engine as _Engine
_model, _tok, _ = _load_model(EVAL_SOURCE, torch.device("cuda"), phase="eval")
_engine = _Engine(_model, _tok)
_stop = {_tok.encode_special("<|assistant_end|>"), _tok.get_bos_token_id()}
def run_cli(prompt, max_tokens=3000):
    """Same output shape as chat_cli ("Assistant: <answer>"), temperature 0.2, without reloading the model."""
    ids = [_tok.get_bos_token_id(), _tok.encode_special("<|user_start|>")] + _tok.encode(prompt) + \
          [_tok.encode_special("<|user_end|>"), _tok.encode_special("<|assistant_start|>")]
    out = []
    for col, _ in _engine.generate(ids, num_samples=1, max_tokens=max_tokens, temperature=0.2, top_k=50):
        if col[0] in _stop:
            break
        out.append(col[0])
    return "Assistant: " + _tok.decode(out)
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
    out = run_cli(row["prompt"])
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
    out = run_cli(row["prompt"])
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
print("\\n=== Real-pattern eval: unseen long, multi-toolkit flows (closest thing to real flows) ===")
sys.path.insert(0, "/content/ooa")
import ooa_reward as RW
os.environ["OOA_CATALOG"] = f"{DRIVE_DIR}/sft/catalog.jsonl"; os.environ["OOA_SCHEMA"] = "/content/platform/contracts/flow-schema/flow.v1.json"
RW.configure()
pr = []
for row in map(json.loads, open(f"{DRIVE_DIR}/sft/eval_pattern.jsonl")):
    out = run_cli(row["prompt"])
    sc = RW.score(out.split("Assistant:", 1)[-1], RW.gold_actions(row["gold_flow"]))
    pr.append(sc)
print(f"real-pattern eval: {len(pr)} | valid (passes every rule) {sum(x['valid'] for x in pr)} | exact action sequence {sum(x['solved'] for x in pr)} | mean score {sum(x['total'] for x in pr)/max(1,len(pr)):.3f}")
def _in_order(need, got):
    it = iter(got)
    return all(any(a == g for g in it) for a in need)
_sets = [("plain_requests.jsonl", "set 1: 20 everyday requests (hand-written)"),
         ("plain_requests_2.jsonl", "set 2: 15 more, never tuned on"),
         ("plain_requests_3.jsonl", "set 3: whole business processes, held out from training")]
_summary = []
for _f, _label in _sets:
    if not os.path.exists(f"/content/ooa/{_f}"):
        continue
    print(f"\\n=== Plain-wording eval, {_label} ===")
    pl = []
    for row in map(json.loads, open(f"/content/ooa/{_f}")):
        out = run_cli(row["prompt"])
        sc = RW.score(out.split("Assistant:", 1)[-1])
        try: got = RW.actions_of(json.loads(RW.clean(out.split("Assistant:", 1)[-1])))
        except Exception: got = []
        pl.append((sc["valid"], sc["valid"] and _in_order(row["must"], got)))
        print(("OK  " if pl[-1][1] else ("VALID" if pl[-1][0] else "FAIL ")), row["prompt"][:70], "->", got[:6])
    _summary.append(f"{_label.split(':')[0]}: {sum(o for _, o in pl)}/{len(pl)} right ({sum(v for v, _ in pl)} pass every rule)")
print("\\nplain-wording eval (bare model) | " + " | ".join(_summary))
if ASSISTANT_EVAL:
    # the same pipeline as the PC app and Studio: the model can only write real action / input names (catalog lock),
    # a loop guard stops repetition, typo-level repair is applied, the platform rules verify, up to 2 retries
    print("\\n=== Full assistant: catalog lock + repair + verify (what users get) ===")
    sys.path[:0] = ["/content/ooa/serving"]
    import ooa_server as _S
    from model_backend import NanochatLocal as _Local
    _local = _Local(os.environ["NANOCHAT_BASE_DIR"], "/content/nanochat", source=EVAL_SOURCE, device="cuda",
                    catalog=f"{DRIVE_DIR}/sft/catalog.jsonl")
    _router = _S.Router(f"{DRIVE_DIR}/sft/catalog.jsonl", "/content/platform",
                        _S.Policy(allow_escalation=False, local_retries=2), _local, None)
    _asum = []
    for _f, _label in _sets:
        if not os.path.exists(f"/content/ooa/{_f}"):
            continue
        print(f"--- {_label}")
        ok = ver = 0; n = 0
        for row in map(json.loads, open(f"/content/ooa/{_f}")):
            n += 1
            res = _router.run(row["prompt"])
            got = RW.actions_of(json.loads(res["flow"])) if res.get("verified") else []
            right = bool(res.get("verified")) and _in_order(row["must"], got)
            ver += bool(res.get("verified")); ok += right
            print(("OK   " if right else ("VALID" if res.get("verified") else "FAIL ")), f"{res.get('seconds', 0):>5}s", row["prompt"][:60], "->", got[:6])
        _asum.append(f"{_label.split(':')[0]}: {ok}/{n} right ({ver} verified)")
    print("\\nplain-wording eval (full assistant) | " + " | ".join(_asum))
    del _local, _router
del _engine, _model                    # free the GPU for later cells (RL, self-improvement)
torch.cuda.empty_cache()
print("\\nTry it yourself:")
!cd /content/nanochat && python -m scripts.chat_cli -i sft -p "Which OOA action would I use to read a CSV file into a table?" -t 0.2
''')

import colab_rl_cells
colab_rl_cells.build(md, code)

import base64, zlib
_frozen = {n: (H / "autoresearch/frozen" / n).read_text(encoding="utf-8") for n in ("frozen_eval.jsonl", "frozen_catalog.jsonl", "frozen_eval.sha256")}
FROZEN_BLOB = base64.b64encode(zlib.compress(json.dumps(_frozen).encode(), 9)).decode()
import colab_assistant_cell
colab_assistant_cell.build(md, code)
import colab_ar_cells
colab_ar_cells.build(md, code, FROZEN_BLOB)

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
