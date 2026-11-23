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
    "rl_setup.py": (H / "rl_setup.py").read_text(encoding="utf-8"),
    "gen_targeted.py": (H / "gen_targeted.py").read_text(encoding="utf-8"),
    "gen_long.py": (H / "gen_long.py").read_text(encoding="utf-8"),
    "gen_clarify.py": (H / "gen_clarify.py").read_text(encoding="utf-8"),
    "gen_beginner.py": (H / "gen_beginner.py").read_text(encoding="utf-8"),
    "eval/beginner_requests.jsonl": (H.parent / "eval/beginner_requests.jsonl").read_text(encoding="utf-8"),
    "eval/plain_requests.jsonl": (H.parent / "eval/plain_requests.jsonl").read_text(encoding="utf-8"),
    "eval/plain_requests_2.jsonl": (H.parent / "eval/plain_requests_2.jsonl").read_text(encoding="utf-8"),
    "eval/plain_requests_3.jsonl": (H.parent / "eval/plain_requests_3.jsonl").read_text(encoding="utf-8"),
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
    "sft_setup.py": (H / "sft_setup.py").read_text(encoding="utf-8"),
    "safety.py": (H / "safety.py").read_text(encoding="utf-8"),
    "model_backend.py": (H.parent / "assistant/model_backend.py").read_text(encoding="utf-8"),
    "constrained.py": (H.parent / "assistant/constrained.py").read_text(encoding="utf-8"),
    "serving/ooa_server.py": (H.parent / "serving/ooa_server.py").read_text(encoding="utf-8"),
    "serving/backends.py": (H.parent / "serving/backends.py").read_text(encoding="utf-8"),
    "flow_check.py": (H.parent / "eval/flow_check.py").read_text(encoding="utf-8"),
    "plain_requests.jsonl": (H.parent / "eval/plain_requests.jsonl").read_text(encoding="utf-8"),
    "plain_requests_2.jsonl": (H.parent / "eval/plain_requests_2.jsonl").read_text(encoding="utf-8"),
    "plain_requests_3.jsonl": (H.parent / "eval/plain_requests_3.jsonl").read_text(encoding="utf-8"),
    "beginner_requests.jsonl": (H.parent / "eval/beginner_requests.jsonl").read_text(encoding="utf-8"),
    "beginner_score.py": (H.parent / "eval/beginner_score.py").read_text(encoding="utf-8"),
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
SFT_SEQ = 4096             # context window in tokens for SFT (was 2048): long flows fit whole instead of in parts
SFT_DEVICE_BATCH = 2       # rows per GPU step; 2 x 4096 = the same tokens as 4 x 2048. If the T4 runs out of memory, use 1
ROW_CHARS = 6500 * SFT_SEQ // 2048    # longest training row in characters (filter and render)
ROW_TOKENS = SFT_SEQ - 28             # longest training row in estimated tokens (render)
SFT_EPOCHS = 2 if PROFILE == "smoke" else 6      # only for the old cells 10 and 12 (cell 7 now plans the mixture by budget)
NATURAL_REPEAT = 5         # passes over sft/natural_train.jsonl per epoch, if that file is in Drive (everyday wording)
CORE_REPEAT = 5            # passes over sft/core_train.jsonl per epoch, if in Drive (everyday requests for the 65 most used actions)
REBUILD_DATA = False       # cell 6 reuses the generated flows already in Drive (saves time); True = generate them again
FLOW_REPEAT = 5            # real flows repeated per pass; was 15, which made the model recite whole training flows (memorising)
SYNTH_TRAIN = 1500 if PROFILE == "smoke" else 30000   # generated flows (no LLM), see gen_flows.py
SYNTH_EVAL = 20            # unseen generated flows scored by exact action sequence
SYNTH_PATTERN = 2000 if PROFILE == "smoke" else 12000   # long, multi-toolkit flows following real-flow patterns (gen_flows.real_patterns)
# Reinforcement layer (cell 9: RL with the platform checker as the reward, in the current plan + compact-flow format)
RL_V5 = False              # False until the v5 SFT model (cell 7) has finished and cell 8 has scored it. Then set True and
                           # run cells 9, 9b and 8: RL continues that SFT model (chatrl_checkpoints) and cell 8 tests the RL
                           # model; keep it only if cell 8 scores it better than the SFT model
RL_STEPS = 20 if PROFILE == "smoke" else 200       # optimizer steps: estimated about 3 h per 100 steps on a T4 (range
                           # 2-6.5 h; cell 9 prints it), saved every RL_EVERY steps and resumed after a disconnect
RL_PROMPTS = 60 if PROFILE == "smoke" else 2400    # fresh requests with a gold answer (never in SFT data or tests)
RL_SAMPLES = 8             # answers per request, scored by the checker; the group's mean is the baseline
RL_EXAMPLES_PER_STEP = 2 if PROFILE == "smoke" else 8   # requests per optimizer step
RL_DEVICE_BATCH = 2        # answers per training pass (2 x 4096 tokens, as SFT); 1 if the T4 runs out of memory
RL_MAX_NEW = 1536          # new tokens per answer at most (the window's rest when less): the longest gold answers ~1,100
RL_EVERY = 10 if PROFILE == "smoke" else 25        # evaluate on the held-out RL requests and save every N steps
RL_LR_FRAC = 0.02          # learning rate as a share of nanochat's base rates (its RL default 0.05 moves a 75M model fast)
RL_SEED = 1123             # generator seed of the RL requests (the SFT data used 1-41)
REBUILD_RL_DATA = False    # True: make the RL requests again (they are rebuilt anyway when sft_render changes)
SELF_IMPROVE_ROUNDS = 2
SI_PROMPTS = 20 if PROFILE == "smoke" else 100     # fresh requests per self-improvement round
SI_SAMPLES = 8
EVAL_SOURCE = "sft"        # which model the test cell loads: "sft" or "rl"
SFT_ITERS = 400 if PROFILE == "smoke" else 48000           # counted in MICRO-batches (4 per optimizer step): 48000 = 12000 steps, about 3.5-4.5 h on a T4 (v4: 8000 steps in about 2.5-3 h)
SFT_STEPS = SFT_ITERS // 4                                   # optimizer steps (--total-batch-size = 4 micro-batches)
ANSWER_FORMAT = "compact"  # cell 6d: answers carry only what the model decides (no variables list, step ids or descriptions);
                           # the server and cell 8 rebuild the full flow (ooa_format.expand_flow). "full" = the old answers
GOLD_DROPOUT = 0.5         # cell 6d: share of flow rows whose context misses 20-45% of the needed actions (retrieval misses
                           # them at serving too); 0 = the needed actions are always listed (the old data)
OLD_FORMAT_CELLS = False   # cells 10-13 (self-improvement, autoresearch, whole-assistant demo) still use plain prompts and
                           # plain-JSON answers capped at 1024 tokens: they must not run against a model trained on the new
                           # format, so they are off. True runs them anyway (only for a model trained on the old format)
CLEAN_DATA = True          # cell 6 runs the strict data filter (sft/ -> sft_clean/) and cell 7 trains on the cleaned data
SCEN_REPEAT = 15           # whole business processes are long and few: weight them 3x the other files (CORE_REPEAT)
RENDER = True              # cell 6d writes the data in the plan-first format with retrieval context (sft_render/); cell 7 trains on it
NEW_FILES = {"long_train.jsonl": 15, "business2_train.jsonl": 15, "targeted_train.jsonl": 8,   # new files in Drive sft/
             "fix_train.jsonl": 5, "clarify_train.jsonl": 5,                                     # -> times per epoch
             "beginner_train.jsonl": 4,   # beginner wording: answers start "Clear request: ..." (gen_beginner.py)
             "v5_train.jsonl": 5}         # v5: business words -> steps, which app / action, natural Hinglish & voice, asking
ASSISTANT_EVAL = True      # cell 8 also scores the full assistant (real action names only, repair, loop guard), as users get it
MODE_EVAL = True           # cell 8 also scores the assistant's modes (fast / smart / ultra / auto: serving/ooa_server.MODES)
SMOLTALK_ROWS = 3000       # a little general chat data mixed in so the model still talks; 0 = OOA data only
MAX_FLOW_CHARS = 9000 * SFT_SEQ // 2048   # skip training flows longer than this (about SFT_SEQ tokens)
print(PROFILE, P, "iterations:", NUM_ITERS)
print("SFT settings -> SYNTH_TRAIN:", SYNTH_TRAIN, "| SFT steps:", SFT_STEPS, f"({SFT_ITERS} micro-batches)", "| window:", SFT_SEQ,
      "| answers:", ANSWER_FORMAT, "| gold dropout:", GOLD_DROPOUT, "| CLEAN_DATA:", CLEAN_DATA)
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

# nanochat at the pinned commit; a folder left half-downloaded by an earlier failed clone is removed and cloned again
import shutil
if not (os.path.exists("/content/nanochat/.git") and os.path.exists("/content/nanochat/scripts/chat_sft.py")):
    if os.path.exists("/content/nanochat"):
        print("nanochat folder is incomplete - downloading it again")
        shutil.rmtree("/content/nanochat", ignore_errors=True)
    !git clone -q https://github.com/karpathy/nanochat /content/nanochat
!cd /content/nanochat && (git cat-file -e {NANOCHAT_COMMIT} 2>/dev/null || git fetch -q origin) && git checkout -q -f {NANOCHAT_COMMIT} && git log -1 --oneline
assert os.path.exists("/content/nanochat/tasks") and os.path.exists("/content/nanochat/scripts/chat_sft.py"), \
    "nanochat did not download (GitHub not reachable?). Run this cell again; if it repeats, send the output."
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
import json, os
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
    !python /content/ooa/data_filter.py {DRIVE_DIR}/sft {DRIVE_DIR}/sft_clean /content/platform /content/ooa --cap 60 --max-chars {ROW_CHARS}
''' % json.dumps(files))

code('''
#@title 6d. Write the data in the plan-first format (new): sft_clean + the new files -> sft_render
import os, json
# Every request gets a "### Context" block (the likely actions, 0-2 similar examples, sometimes the agent's variables)
# and every answer becomes "Plan: 1. ... 2. ..." then "Flow: {...}". Flows too long for the window are written as their
# parts (window = SFT_SEQ tokens). Variables are written $name$ as the platform now saves them. About 10-20 minutes
# (per-clause retrieval votes for long requests are the slow part).
# ANSWER_FORMAT "compact": the flow after "Flow:" has no variables list, step ids or descriptions (about 45% fewer answer
# tokens on long flows); code rebuilds them. GOLD_DROPOUT: half the flow rows miss 20-45% of the needed actions in the
# context, as at serving, so the model learns to write an action it knows when it is not listed.
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
    !cd /content/ooa && python render_dataset.py {_src} {DRIVE_DIR}/sft_render --catalog {DRIVE_DIR}/sft/catalog.jsonl --max-chars {ROW_CHARS} --max-tokens {ROW_TOKENS} --answer-format {ANSWER_FORMAT} --gold-dropout {GOLD_DROPOUT} --extra {_x}
    # catalog question rows stay as they are
    shutil.copy(f"{_src}/train_catalog.jsonl", f"{DRIVE_DIR}/sft_render/train_catalog.jsonl")
    _r = json.load(open(f"{DRIVE_DIR}/sft_render/render_report.json"))
    print("rendered from", _src, "+", len(_extra), "new files ->", f"{DRIVE_DIR}/sft_render")
    print(json.dumps({k: v for k, v in _r.items() if not isinstance(v, (dict, list))}, indent=1))
''')

code('''
#@title 7. Patch nanochat SFT (data mix by step budget, train with the 4096 window), then train
import os, sys, shutil, time, importlib
sys.path.insert(0, "/content/ooa")
import sft_setup as _T
importlib.reload(_T)
SFT_DIR = f"{DRIVE_DIR}/sft_clean" if CLEAN_DATA and os.path.exists(f"{DRIVE_DIR}/sft_clean/filter_report.json") else f"{DRIVE_DIR}/sft"
if RENDER and os.path.exists(f"{DRIVE_DIR}/sft_render/render_report.json"):
    SFT_DIR = f"{DRIVE_DIR}/sft_render"
print("training data:", SFT_DIR)
# always start from the pinned nanochat scripts, so an earlier patch can never hide the current data mix: write them
# straight from the pinned commit (independent of the working tree, the index or an earlier edit of this cell)
!cd /content/nanochat && git fetch -q origin 2>/dev/null; git cat-file -e {NANOCHAT_COMMIT} && git show {NANOCHAT_COMMIT}:scripts/chat_sft.py > scripts/chat_sft.py && git show {NANOCHAT_COMMIT}:scripts/chat_cli.py > scripts/chat_cli.py && echo "chat_sft.py and chat_cli.py written from the pinned nanochat {NANOCHAT_COMMIT}"
!cp /content/ooa/ooa_task.py /content/nanochat/tasks/ooa.py
# The data mix is planned for the real budget (SFT_STEPS optimizer steps x tokens per step), not "epochs x repeats":
# every file gets a number of views per row; long / business rows at most 3, catalog questions at least 3.
_tps = SFT_SEQ * SFT_DEVICE_BATCH * 4
VIEWS, _stats, _budget, _lines = _T.plan_for_notebook(SFT_DIR, SFT_STEPS, _tps, smoltalk_rows=SMOLTALK_ROWS)
print(f"SFT budget: {SFT_STEPS} steps x {_tps:,} tokens per step -> about {_budget/1e6:.0f}M tokens of conversations"
      f" (+ {SMOLTALK_ROWS} SmolTalk rows). Planned views per row:")
print("\\n".join(_lines))
# The patch (sft_setup.patch_chat_sft; it stops if nanochat changed upstream):
#  - the attention window: nanochat builds the SFT model from the BASE config (sequence 1024 on the t4 profile), so SFT
#    used to train with a 1024 window while the checkpoint said 4096. Now the window is --max-seq-len (printed at the
#    start as "SFT attention window: ...") and the checkpoint saves "trained_window" for the serving check.
#  - the mix above; conversations longer than the window are skipped, never cut; at the end it prints how many rows
#    of each file were actually read ("Training data actually read").
p = "/content/nanochat/scripts/chat_sft.py"
_s = open(p).read().replace("\\r\\n", "\\n")
_need = {"window": _T.WINDOW_ANCHOR, "meta": _T.META_ANCHOR, "tasks start": _T.TASKS_START, "tasks end": _T.TASKS_END,
         "import": _T.IMPORT_ANCHOR, "loader": _T.LOADER_OLD, "step": _T.STEP_ANCHOR, "end": _T.END_ANCHOR}
_miss = [k for k, a in _need.items() if a not in _s]
if _miss:      # say exactly what differs, so it can be fixed from this output
    print("chat_sft.py does not match the pinned nanochat; missing:", _miss, "| sha1 of the file:",
          __import__("hashlib").sha1(_s.encode()).hexdigest()[:12])
    for _i, _l in enumerate(_s.splitlines()):
        if any(w in _l for w in ("torch.compile", "orig_model =", "user_config\\": ", "train_tasks =", "render_conversation(")):
            print(f"  line {_i + 1}: {_l!r}")
    !cd /content/nanochat && git log -1 --oneline && git status --short | head
    raise SystemExit("cell 7 stopped before training: send the lines above")
open(p, "w").write(_T.patch_chat_sft(_s, SFT_DIR, VIEWS, smoltalk_rows=SMOLTALK_ROWS))
print("patched chat_sft.py")
# keep the previous model: nanochat loads the highest step in chatsft_checkpoints, so move it aside (never deleted)
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
import os, subprocess, json, sys
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
_model, _tok, _meta = _load_model(EVAL_SOURCE, torch.device("cuda"), phase="eval")
_engine = _Engine(_model, _tok)
_stop = {_tok.encode_special("<|assistant_end|>"), _tok.get_bos_token_id()}
from model_backend import window_check as _window_check, base_window as _base_window
_tw, _ww = _window_check(_meta, _base_window(os.environ["NANOCHAT_BASE_DIR"]))
print("attention window: served", _model.config.sequence_len, "| trained", _tw, "|", _ww or "OK (trained = served)")
_last = {"ended": False, "tokens": 0, "json_stopped": False}
def run_cli(prompt, max_tokens=3000):
    """Same output shape as chat_cli ("Assistant: <answer>"), temperature 0.2, without reloading the model. This is the
    BARE model (no loop guard, no catalog lock): _last says whether it ended on its own and how many tokens it wrote."""
    ids = [_tok.get_bos_token_id(), _tok.encode_special("<|user_start|>")] + _tok.encode(prompt) + \
          [_tok.encode_special("<|user_end|>"), _tok.encode_special("<|assistant_start|>")]
    out, ended = [], False
    _cap = max(64, min(max_tokens, _model.config.sequence_len - len(ids)))     # never past the window
    for col, _ in _engine.generate(ids, num_samples=1, max_tokens=_cap, temperature=0.2, top_k=50):
        if col[0] in _stop:
            ended = True
            break
        out.append(col[0])
    _last.update(ended=ended, tokens=len(out))
    return "Assistant: " + _tok.decode(out)
# the new model (trained on sft_render) reads each request with a "### Context" block and answers "Plan: ... Flow: {...}";
# the prompt is built exactly as the server and Studio build it (serving/ooa_server.Router.prompt)
sys.path[:0] = ["/content/ooa", "/content/ooa/serving"]
import ooa_format as _F, ooa_server as _S
NEW_FORMAT = globals().get("RENDER", False) and os.path.exists(f"{DRIVE_DIR}/sft_render/examples.jsonl")
_EV_DIR = f"{DRIVE_DIR}/sft_render" if NEW_FORMAT else f"{DRIVE_DIR}/sft"
_EV_EXAMPLES = f"{_EV_DIR}/examples.jsonl" if NEW_FORMAT else None
_prompter = _S.Router(f"{_EV_DIR}/catalog.jsonl", "/content/platform", _S.Policy(allow_escalation=False), None, None,
                      _EV_EXAMPLES, "context" if NEW_FORMAT else "plain")
print("prompt format:", "context + plan (new)" if NEW_FORMAT else "plain request (old model)")
_kinds = {}
_cat_entries = {json.loads(l)["id"]: json.loads(l) for l in open(f"{_EV_DIR}/catalog.jsonl")}
def ask(request):
    """The model's flow JSON text for a plain request (plan stripped); questions / replies give their text. A compact
    answer (a model trained with ANSWER_FORMAT "compact") is expanded into the full flow the way the server does it
    (ooa_format.expand_flow); a full answer (older models) is used as written. _last["json_stopped"]: the answer ended
    on its own and its flow is one valid JSON object."""
    text = run_cli(_prompter.prompt(request) if NEW_FORMAT else request)
    text = text.split("Assistant:", 1)[-1].split("<|assistant_end|>")[0].strip()
    if not NEW_FORMAT:
        _last["json_stopped"] = _last["ended"] and _F.flow_object(text) is not None
        return text
    ans = _F.parse_answer(text, expand=True, catalog=_cat_entries, request=request)
    _kinds[ans["kind"]] = _kinds.get(ans["kind"], 0) + 1
    if ans.get("expanded"):
        _kinds["expanded (compact answers)"] = _kinds.get("expanded (compact answers)", 0) + 1
    _last["json_stopped"] = (_last["ended"] and ans["kind"] == "flow"
                             and _F.flow_object(ans.get("compact_text") or ans.get("flow_text")) is not None)
    return ans.get("flow_text") or ans.get("text") or ""
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
    text = ask(row["prompt"])
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
    text = ask(row["prompt"])
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
os.environ["OOA_CATALOG"] = f"{_EV_DIR}/catalog.jsonl"; os.environ["OOA_SCHEMA"] = "/content/platform/contracts/flow-schema/flow.v1.json"
RW.configure()
pr = []
for row in map(json.loads, open(f"{DRIVE_DIR}/sft/eval_pattern.jsonl")):
    sc = RW.score(ask(row["prompt"]), RW.gold_actions(row["gold_flow"]))
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
        _txt = ask(row["prompt"])
        sc = RW.score(_txt)
        try: got = RW.actions_of(json.loads(RW.clean(_txt)))
        except Exception: got = []
        pl.append((sc["valid"], sc["valid"] and _in_order(row["must"], got), _last["json_stopped"], _last["ended"], _last["tokens"]))
        print(("OK  " if pl[-1][1] else ("VALID" if pl[-1][0] else "FAIL ")), f"{_last['tokens']:>5} tok",
              "ended" if _last["ended"] else "CUT  ", row["prompt"][:56], "->", got[:6])
    _summary.append(f"{_label.split(':')[0]}: {sum(x[1] for x in pl)}/{len(pl)} right ({sum(x[0] for x in pl)} pass every rule, "
                    f"{sum(x[2] for x in pl)} valid JSON and stopped, {sum(not x[3] for x in pl)} cut at the token cap, "
                    f"median {sorted(x[4] for x in pl)[len(pl) // 2]} answer tokens)")
print("\\nplain-wording eval (BARE model: raw generation, no loop guard, no catalog lock, no repair) | " + " | ".join(_summary))
if NEW_FORMAT:
    print("answer kinds (flow / question / reply):", _kinds)
# Beginner set: 80 requests the way people with no IT background write them (vague goals, business words, casual,
# typos, Hinglish, Indian English, voice, story). 61 must be built (right = passes every rule + has the must actions
# in order; a must entry may list ids that do the same job), 19 point at something only the user knows and must be
# ASKED about (right = a Question naming that detail; a flow built on a guess counts as "assumed", not right). Scored
# per kind and per style (eval/beginner_score.py).
_BEG = "/content/ooa/beginner_requests.jsonl"
if NEW_FORMAT and os.path.exists(_BEG):
    import beginner_score as _B
    _brows = _B.load(_BEG)
    print("\\n=== Beginner requests, BARE model ===")
    _bres, _bclear = [], 0
    for row in _brows:
        _txt = run_cli(_prompter.prompt(row["prompt"])).split("Assistant:", 1)[-1].split("<|assistant_end|>")[0].strip()
        _a = _F.parse_answer(_txt, expand=True, catalog=_cat_entries, request=row["prompt"])
        _bclear += bool(_a.get("clear"))
        _valid, _acts = False, []
        if _a["kind"] == "flow" and _a["flow_text"]:
            _valid = bool(RW.score(_a["flow_text"])["valid"])
            try: _acts = RW.actions_of(json.loads(RW.clean(_a["flow_text"])))
            except Exception: _acts = []
        _r = _B.score(row, _a["kind"], _valid, _acts, _a.get("question"))
        _bres.append(_r)
        print(("OK   " if _r["right"] else _r["outcome"][:12].ljust(12)), row["kind"][:4], ",".join(row["style"])[:22].ljust(22),
              row["prompt"][:48], "->", (_a.get("question") or _a.get("clear") or str(_acts[:4]))[:70])
    print(f"beginner set (BARE model) | {_B.summary_text(_B.summarize(_brows, _bres))} | answers restating the request: {_bclear}")
if ASSISTANT_EVAL:
    # the same pipeline as the PC app and Studio: the model can only write real action / input names (catalog lock),
    # a loop guard stops repetition, typo-level repair is applied, the platform rules verify, up to 2 retries
    print("\\n=== Full assistant: catalog lock + repair + verify (what users get) ===")
    sys.path[:0] = ["/content/ooa/serving"]
    import ooa_server as _S
    from model_backend import NanochatLocal as _Local
    _local = _Local(os.environ["NANOCHAT_BASE_DIR"], "/content/nanochat", source=EVAL_SOURCE, device="cuda",
                    catalog=f"{_EV_DIR}/catalog.jsonl")
    # the same Router as Studio: context + plan, plan coverage, self-correction with the problems, parts for long flows
    _router = _S.Router(f"{_EV_DIR}/catalog.jsonl", "/content/platform",
                        _S.Policy(allow_escalation=False, local_retries=2), _local, None,
                        _EV_EXAMPLES, "context" if NEW_FORMAT else "plain")
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
    if NEW_FORMAT and os.path.exists(_BEG):
        print("\\n=== Beginner requests, full assistant ===")
        _bres2 = []
        for row in _brows:
            res = _router.run(row["prompt"])
            _r = _B.score_router(row, res, lambda t: RW.actions_of(json.loads(t)))
            _bres2.append(_r)
            got = RW.actions_of(json.loads(res["flow"])) if res.get("verified") else []
            print(("OK   " if _r["right"] else _r["outcome"][:12].ljust(12)), f"{res.get('seconds', 0):>5}s", row["kind"][:4],
                  row["prompt"][:48], "->", (res.get("question") or res.get("understood") or str(got[:4]))[:70])
        print("beginner set (full assistant) | " + _B.summary_text(_B.summarize(_brows, _bres2)))
    if MODE_EVAL and NEW_FORMAT:
        # the modes Studio's Mode menu offers (the agent team: planner, parallel builders in one batch, fixer, critic)
        print("\\n=== Modes: right / verified / mean seconds per request ===")
        _mrows = [(_f, _label, [json.loads(l) for l in open(f"/content/ooa/{_f}")]) for _f, _label in _sets
                  if os.path.exists(f"/content/ooa/{_f}")]
        if os.path.exists(_BEG):
            _mrows.append(("beginner_requests.jsonl", "beginner", _brows))
        for _mode in ("fast", "smart", "ultra", "auto"):
            _line = []
            for _f, _label, _rows in _mrows:
                ok = ver = 0; _secs = []
                for row in _rows:
                    res = _router.run(row["prompt"], mode=_mode)
                    _secs.append(res.get("seconds", 0))
                    if "kind" in row:
                        right = _B.score_router(row, res, lambda t: RW.actions_of(json.loads(t)))["right"]
                    else:
                        got = RW.actions_of(json.loads(res["flow"])) if res.get("verified") else []
                        right = bool(res.get("verified")) and _in_order(row["must"], got)
                    ok += right; ver += bool(res.get("verified"))
                _line.append(f"{_label.split(':')[0]}: {ok}/{len(_rows)} right, {ver} verified, "
                             f"{sum(_secs) / max(1, len(_secs)):.1f} s")
            print(f"{_mode:6} | " + " | ".join(_line))
    del _local, _router
del _engine, _model                    # free the GPU for later cells (RL, self-improvement)
torch.cuda.empty_cache()
print("\\nTry it yourself:")
!cd /content/nanochat && python -m scripts.chat_cli -i sft -p "Which OOA action would I use to read a CSV file into a table?" -t 0.2
''')

def gated(t):
    """A cell that runs only when OLD_FORMAT_CELLS is True (cell 1): cells 10-13 still send plain prompts and read plain
    JSON answers capped at 1024 tokens, so against a model trained on the context + plan (+ compact) format they would
    score and train on the wrong thing. Off by default, with a clear note."""
    lines = t.strip("\n").splitlines()
    head = [lines[0]] if lines and lines[0].startswith("#@title") else []
    body = lines[len(head):]
    note = ('print("Skipped (OLD_FORMAT_CELLS = False in cell 1): this cell still uses plain prompts and plain-JSON answers '
            'capped at 1024 tokens, but the model from cells 6d-7 is trained on the context + plan format with compact '
            'answers. It must not run against that model. Set OLD_FORMAT_CELLS = True only for a model trained on the '
            'old format.")')
    code("\n".join(head + ["if not globals().get(\"OLD_FORMAT_CELLS\", False):", "    " + note, "else:"]
                   + [("    " + l) if l.strip() else l for l in body]))


import colab_rl_cells
colab_rl_cells.build(md, code, gated)

import base64, zlib
_frozen = {n: (H / "autoresearch/frozen" / n).read_text(encoding="utf-8") for n in ("frozen_eval.jsonl", "frozen_catalog.jsonl", "frozen_eval.sha256")}
FROZEN_BLOB = base64.b64encode(zlib.compress(json.dumps(_frozen).encode(), 9)).decode()
import colab_assistant_cell
colab_assistant_cell.build(md, gated)
import colab_ar_cells
colab_ar_cells.build(md, gated, FROZEN_BLOB)

md("""
## After this run
- Checkpoints: `DRIVE_DIR/nanochat_base/base_checkpoints` (pretrained) and `chatsft_checkpoints` (OOA-trained).
- Weak results on the held-out test are expected with the seed data. The next lever is more and better dialogues
  (edits, error fixing, missing-input questions), validated with the scorer in `eval/flow_check.py`.
- Reinforcement layer: cell 9 (RL with the platform checker as the reward, in the current format; `RL_V5` in cell 1,
  off until the v5 SFT model is trained and scored). Self-improvement (10), autoresearch (12) and the whole-assistant
  demo (13) still use plain prompts and plain-JSON answers, so they are off (`OLD_FORMAT_CELLS = False`)
  for models trained on the context + plan format. The full assistant is tested in cell 8. Feed real usage back with `ingest_feedback.py`.
- For a stronger base, use `PROFILE = "t4"` and let it run (resume after disconnects), or an A100 on Colab Pro.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 0}
out = H / "ooa_nanochat_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
