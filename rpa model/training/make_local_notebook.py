"""Generate ooa_local_pc.ipynb: train the OOA model on a Windows/Linux PC with the Colab v4 recipe, test it the way
Colab cell 8 does, install it only if it is better, and run the assistant. No Colab needed.
Run:  python make_local_notebook.py [DRIVE_FOLDER_ID] [OUT.ipynb]
      (default output "rpa model/ooa_local_pc.ipynb"; the folder id is never stored in the repo: leave it out for the
      shared notebook, and pass it only for a personal copy written outside the repo)
The notebook sits in the "rpa model" folder of ooa_assistant.zip and drives training/local/{data_pc,train_pc,eval_pc}.py,
which reuse the Colab modules (sft_setup.py, render_dataset.py, data_filter.py, ooa_format.py).
"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
if any(x.startswith("-") for x in sys.argv[1:]):                 # e.g. --help: never written into the notebook
    raise SystemExit(__doc__)
FOLDER_ID = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else H.parent / "ooa_local_pc.ipynb"
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})

md("""
# OOA model on this PC: download, train (Colab v4 recipe), test, install, run

Run the cells from top to bottom (Shift+Enter): **1 settings -> 2 check this PC -> 3 download -> 4 check -> 5 train ->
6 test and install -> 7 start the assistant**. Works on Windows and Linux, with or without an NVIDIA graphics card.
The training is the same code as the Colab notebook (cell 7): nanochat's SFT script from the pinned commit, patched by
`training/sft_setup.py` (the 4096-token attention-window fix, the data mixture planned for the step budget), on the
same data (`sft_render`: plan-first answers, compact flows, gold dropout, beginner rows).

| `MODE` (cell 1) | What it does | Time |
|---|---|---|
| `"auto"` (default) | picks for you and says why in cell 3: NVIDIA GPU + a finished trained model on Drive or this PC -> `"refine"`; NVIDIA GPU and no trained model anywhere -> `"full"`; no GPU -> `"topup"` | |
| `"refine"` (NVIDIA GPU) | **continues the finished Colab model** (v5; or this PC's model when it already grew from it) for `REFINE_STEPS` (1500) steps, saved every 250 steps: about 80% of the training goes to the newest rows (v5, session: sign in once and reuse it) and the weak areas (beginner wording, long business processes, clarify, targeted, fix, your feedback), about 18% replays everything else so nothing is forgotten. Learning rate 0.35 (a gentle continue, not a restart) | estimated 40-65 min on a T4-class GPU (RTX 3060/4060; the Colab T4 with torch.compile about 30-35 min), plus 1-2 min of loading per chunk |
| `"full"` (NVIDIA GPU) | **the Colab run again**: 12000 steps at the 4096 window from the pretrained **base** model (`BASE = "current"`: the Colab base; `"v6"`: the clean base from `"pretrain"`, with its own tokenizer and its own run state `local_run/full_v6`). It does NOT use the Colab model: only for a fresh start | estimated 3-4.5 h on an RTX 3060/4060 (the Colab T4 takes about 3.5-4.5 h); a CPU would need weeks, so it is not offered there |
| `"topup"` (no GPU) | **continues the newest trained model** (Drive's, or this PC's own when it already grew from it) for `TOPUP_STEPS` steps on the same data, mostly the beginner and long business-process rows | 300 steps: about 5.5-9 h on 4 CPU cores (measured on a 4-core test machine at the 4096 window: 67 s per step at low priority, 55 s with all cores; slower PCs up to about 100 s) |
| `"install"` | no training: tests Drive's newest model against this PC's and installs it if it is better | the test only (cell 6) |
| `"pretrain"` (NVIDIA GPU) | **the clean base model (v6)**, pretrained here from zero on commercially usable text only: FineWeb-Edu (ODC-By 1.0) + 15% of the platform's own docs, code, catalog and the rendered training rows; no ClimbMix (non-commercial). Builds the data (streamed from Hugging Face, about 3 GB), trains its own tokenizer (16,384 tokens; flow JSON takes about half the tokens of the old one), pretrains d8 at a 2048 window on 1.5B tokens, saved every 200 steps (about 22 min) and resumed after any stop; NaN guard. The result is a separate pack, `train_data/base_v6`: the current base and the installed model are not touched. Then `MODE = "full"` with `BASE = "v6"` trains the OOA model on it (cell 6 installs it only if it beats the installed one) | estimated 11-14 h on an RTX 4060 Laptop (8 GB), plus about 10 min for the data; `PRETRAIN_MEASURE = 30` measures this PC's real speed first (2 min) |
| `"rl"` (NVIDIA GPU only) | **reinforcement learning** on the newest trained model (after a refine: the refined model once cell 6 installed it), as Colab cell 9: the model writes 8 answers per fresh request, the platform checker scores them (`training/rl_setup.py`), better answers are reinforced; then cell 6 tests it and installs it only if it is better. Use it after a refine (or on Drive's v5) | `RL_STEPS` 200: estimated about 6 h on an RTX 3060/4060 class GPU (about a T4; range 4-13 h); saved every 25 steps, resumable |

### The clean base model (v6): pretrain it here, then train the OOA model on it

The current base model was pretrained on ClimbMix, which is licensed for non-commercial use only. The v6 base uses
only FineWeb-Edu (ODC-By 1.0: commercial use allowed with attribution) and the platform's own text
(`training/pretrain/DATA_CARD.md`). It is a separate model with its own tokenizer; nothing of the current one changes.

1. Cell 1: `MODE = "pretrain"` (optionally `PRETRAIN_MEASURE = 30` first: cell 5 then only measures the real speed
   and prints the hours; set it back to 0 after). Run cells 1-5. Cell 3 downloads the rendered data (the domain text);
   cell 5 builds the data (about 10 min, about 3 GB), trains the tokenizer (a few minutes) and pretrains (about
   11-14 h on an RTX 4060 Laptop). Keep the laptop plugged in. After any stop run cells 1-5 again: it resumes from the
   newest checkpoint (every 200 steps, about 22 min). It ends with `PRETRAINED: ...train_data/base_v6`.
2. Cell 1: `MODE = "full"`, `BASE = "v6"`. Run cells 1-6: the OOA training on the new base (about 3.5-4.5 h), then
   cell 6 tests it and installs it only if it beats the installed model (with its own tokenizer; the old model and
   tokenizer are kept in `nanochat_base/old_model_<date>`).
3. Optional: `MODE = "refine"` / `"rl"` continue whichever model is installed, as before.

### After a Colab run (the Colab model is trained: refine it here)

1. Cell 1: `MODE = "refine"` (or leave `"auto"`: with an NVIDIA GPU and the Colab model on Drive it picks refine).
2. Run cells 1, 2, 3, 4, 5, 6 in order. Cell 3 downloads the Colab model and its data (and adds the session rows if
   Drive does not have them yet); cell 5 refines (about an hour); cell 6 tests.
3. Look at the score lines at the end of cell 6: the refined model is installed only if it scores more right answers
   than the Colab model (and than the model this PC had). If not, the Colab model is installed when it beats the old one.
4. Optional, for more: `MODE = "rl"` in cell 1, run cells 1-6 again (RL starts from the newest installed model;
   about 6 h). Or set `REFINE_THEN_RL = True` before step 2 and cell 6 runs RL and its test right after the refine.
5. Cell 7 starts the assistant.

Training is saved in chunks (top-up: every 50 steps; refine: every 250; full: one run like Colab, or `FULL_CHUNKS` pieces): after a stop,
restart or crash run cells 1-5 again and it continues from the last saved chunk. On a CPU it runs at low priority with
one core left free, so the PC stays usable.

Broken weights: before training, cell 4 checks that the start model's weights are all numbers. A model with nan
(a run that went unstable, e.g. on Colab) is refused with the setting to use instead: `REFINE_FROM = "pc"` (this PC's
installed model) or `"drive"` (Drive's newest) in cell 1; nothing switches by itself. While cell 5 trains, a loss
that stays nan (5 steps) or a nan validation stops the chunk before it is saved, with a STOPPED line saying what to
send; the next try of that chunk starts with a fresh optimizer state. Cell 6 never installs a model with nan weights.

Cell 6 scores the new model and the installed one through the full assistant (catalog lock, repair, verify,
self-correction), on sets 1-3 and the beginner set, as Colab cell 8 does; on a CPU a sample (`EVAL_PER_SET` requests
per set, at most `EVAL_MINUTES` per model; a request takes 2-70 s on 4 cores, about 15-25 minutes per model). The new
model is **installed only if it is better** (the old one is kept in `nanochat_base/old_model_<date>`), together with
its `catalog.jsonl` and `examples.jsonl` in `rpa model/sft/`, so the assistant gives it the context + plan prompts it
was trained on.

Before the first run: unzip `ooa_assistant.zip`, open this notebook from its `rpa model` folder in VS Code (Python +
Jupyter extensions; also Antigravity) or Jupyter, and pick a 64-bit Python 3.10+ kernel (if the editor asks to
install `ipykernel`, say yes). Everything is installed into that kernel's Python by cell 2; Git is not needed (nanochat
is downloaded as a zip when Git is missing). Internet is needed for cells 2 and 3. The Drive folder `ooa-nanochat`
must be shared as "anyone with the link"; put its id (from the link) in cell 1.
Updating with the code-only zip: rename the old folder (e.g. `rpa model OLD`) and unzip the new one next to it; cell 2
copies the platform files, the installed model, the downloads and your feedback from the old folder.
""")

code(f'''
#@title 1. Settings
DRIVE_FOLDER_ID = "{FOLDER_ID}"     # Google Drive folder ooa-nanochat (shared as "anyone with the link")
MODE = "auto"              # "auto": GPU + a trained model (Drive / this PC) -> "refine"; GPU, no trained model -> "full";
                           # no GPU -> "topup". Or set "refine", "full", "topup", "install", "rl", "pretrain" yourself
BASE = "current"           # MODE "full": the base model the OOA training starts from: "current" (the Colab base) or
                           # "v6" (the clean base: MODE "pretrain" on this PC, or Colab PROFILE "v6"; its own tokenizer)
PRETRAIN_PROFILE = "pc"    # MODE "pretrain" (GPU): d8, window 2048, 1.5B tokens (RTX 4060 8 GB: about 11-14 h)
PRETRAIN_DATA = "build"    # "build": stream FineWeb-Edu and build the shards here (about 10 min, 3 GB); "drive": the
                           # shards on Drive (nanochat_base_v6/base_data_climbmix), when someone uploaded them
PRETRAIN_MEASURE = 0       # > 0: cell 5 only measures this many pretraining steps (tokens/s, memory, real hours)
REFINE_STEPS = 1500        # MODE "refine" (GPU): steps on top of the Colab model (T4-class GPU: about 40-65 min)
REFINE_CHUNKS = 6          # saved every REFINE_STEPS / REFINE_CHUNKS steps (250): after a stop it continues
REFINE_THEN_RL = False     # True: after a refine, cell 6 also runs RL on the newest installed model and tests it
REFINE_FROM = "auto"       # the trained model refine / topup / rl continue: "auto" the newest (Drive's, or this PC's
                           # when it already grew from it); "drive" Drive's newest (the Colab run); "pc" this PC's
                           # installed model. Cell 4 refuses a model with broken weights (nan) and names the other one
RL_STEPS = 200             # MODE "rl" (GPU): RL steps of 8 requests x 8 answers (about 3 h per 100; saved every 25)
TOPUP_STEPS = 300          # top-up length in steps of 32,768 tokens (4 CPU cores: about 5.5-9 h)
TOPUP_HOURS = 0            # > 0: measure this PC's speed first (about 5 min) and size the top-up to this many hours
FULL_STEPS = 12000         # the Colab run (SFT_ITERS 48000 micro-batches = 12000 steps)
FULL_CHUNKS = 1            # 1 = one run exactly as Colab cell 7 (a stop means starting again); e.g. 8 = saved every
                           # 1000 steps and resumable, the same learning-rate schedule (each chunk reloads the data)
SFT_SEQ = 4096             # training window, as Colab; a CPU PC with less than 10 GB RAM uses 2048 (cell 2)
EVAL_PER_SET = 10          # cell 6: requests per test set on a CPU (4 sets; 0 = all 139, GPU default)
EVAL_MINUTES = 60          # cell 6: at most this long per model
REFRESH_DATA = False       # True: download the training data again even if it is unchanged
RESTART_TRAINING = False   # True: throw away a training run in progress and start a new one

RPA_FOLDER = r""           # optional: the 'rpa model' folder, e.g. r"C:/OOA/rpa model" (empty = search for it)

import os, sys, json, shutil, subprocess, time
from pathlib import Path

def _ok(p):
    return (p / "assistant" / "app.py").exists() and (p / "training" / "local" / "train_pc.py").exists()

def _find():
    starts = [Path(RPA_FOLDER)] if RPA_FOLDER else []
    _nb = globals().get("__vsc_ipynb_file__")        # VS Code / Antigravity: this notebook's own path
    if _nb:
        starts += [Path(_nb).parent]
    here = Path.cwd()
    starts += [here, *here.parents]
    home = Path.home()
    starts += [Path("C:/OOA"), home / "Downloads", home / "Desktop", home / "OneDrive" / "Desktop", home / "Documents", here]
    seen = set()
    for s0 in starts:
        for p in [s0, s0 / "rpa model"]:
            try:
                if p in seen or not p.exists():
                    continue
            except OSError:                               # a folder Windows does not let us look into
                continue
            seen.add(p)
            if _ok(p):
                return p
    for s0 in starts[len(starts) - 6:]:          # one more level down, e.g. Downloads/ooa_assistant/rpa model
        if s0.exists():
            try:
                for p in list(s0.glob("*/rpa model")) + list(s0.glob("*/*/rpa model")):
                    if _ok(p):
                        return p
            except OSError:
                pass
    print("Looked in:", *[str(p) for p in seen], sep=chr(10) + "  ")
    raise SystemExit("Could not find the 'rpa model' folder of the NEW ooa_assistant.zip (it must contain "
                     "assistant/app.py and training/local/train_pc.py). Unzip the new zip to C:/OOA, "
                     "or put its path in RPA_FOLDER above, then run this cell again.")

ROOT = _find()
LOCAL = ROOT / "training" / "local"
os.environ["PYTHONUTF8"] = "1"                 # every script started from here reads and writes UTF-8 (Windows)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
print("assistant folder:", ROOT)
print("Python (this notebook's kernel; everything is installed into it):", sys.executable, sys.version.split()[0],
      f"({64 if sys.maxsize > 2**32 else 32}-bit)")
if sys.version_info < (3, 10) or sys.maxsize <= 2**32:
    print("! this Python cannot run the training: it needs a 64-bit Python 3.10 or newer. Install Python 3.12 from "
          "https://www.python.org, then pick it as the kernel (top right of the notebook) and run this cell again.")
''')

code('''
#@title 2. Check this PC and install what is needed (first time: several minutes)
# Every command runs as a list of arguments with this notebook's own Python (no shell lines: on Windows those go to
# cmd.exe), and its output is shown here as it happens.
def stream(cmd, cwd=None):
    \"\"\"Run a step and show its output as it happens; -> exit code. A stopped cell ends the step with everything it
    started (Windows: the whole process tree, since terminating a process there does not end its children).\"\"\"
    print(">", " ".join(str(c) for c in cmd)[:200], flush=True)
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", cwd=None if cwd is None else str(cwd),
                         env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    finally:
        if p.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            else:
                p.terminate()
    return p.wait()

def run(cmd, check=True):
    rc = stream(cmd)
    if check and rc != 0:
        hint = ""
        if os.name == "nt" and "pip" in [str(c) for c in cmd]:
            hint = ("\\nOn Windows the usual causes: the assistant (or another notebook) is running and holds PyTorch's "
                    "files (close it and restart this notebook's kernel), or Windows long paths are off (the error "
                    "names a long path: run PowerShell as administrator, then  New-ItemProperty -Path "
                    "HKLM:\\\\SYSTEM\\\\CurrentControlSet\\\\Control\\\\FileSystem -Name LongPathsEnabled -Value 1 "
                    "-PropertyType DWORD -Force  and restart the PC).")
        raise SystemExit(f"command failed (exit {rc}); see the messages above" + hint)
    return rc

def capture(cmd):
    \"\"\"-> (exit code, stdout, stderr), decoded as UTF-8 whatever the Windows code page; a missing program: 127.\"\"\"
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
        return r.returncode, r.stdout, r.stderr
    except OSError as ex:
        return 127, "", str(ex)

import importlib
sys.path.insert(0, str(LOCAL))
import setup_pc
importlib.reload(setup_pc)
for _p in setup_pc.python_problems():
    raise SystemExit(_p)
for _p in setup_pc.path_notes(ROOT):
    print("note:", _p)

GPU, GPU_MEM = False, 0
_rc, _out, _err = capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
if _rc == 0 and _out.strip():
    try:
        name, mem = _out.strip().splitlines()[0].rsplit(",", 1)
        GPU, GPU_MEM = True, int(float(mem)) // 1024
        print(f"NVIDIA GPU: {name.strip()} ({GPU_MEM} GB)")
    except ValueError:
        pass
if not GPU:
    print("no NVIDIA GPU: training will run on the CPU |", os.cpu_count(), "CPU threads")

if os.name == "nt":                     # Smart App Control blocks PyTorch's files (WinError 4551)
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\\CurrentControlSet\\Control\\CI\\Policy")
        if winreg.QueryValueEx(k, "VerifiedAndReputablePolicyState")[0] == 1:
            raise SystemExit("Windows Smart App Control is ON and will block PyTorch. Turn it off: Windows Security > "
                             "App & browser control > Smart App Control settings > Off, then run this cell again.")
    except OSError:
        pass

# what the code-only update zip does not carry: copied from the older "rpa model" folder next to this one (e.g.
# "rpa model OLD"): platform, the installed model, its context, the downloads, your feedback
_copied = setup_pc.adopt_old(ROOT)
print("copied from the older folder:", ", ".join(_copied) if _copied else "nothing needed")
# nanochat (the model code) at the pinned commit: from the older folder, else git, else GitHub's zip (no Git needed)
print("nanochat:", setup_pc.ensure_nanochat(ROOT), "->", ROOT / "nanochat")
if not setup_pc.platform_ok(ROOT):
    print("! " + setup_pc.PLATFORM_HELP.format(p=ROOT / "platform", rpa=ROOT) + ". Training (cells 3-5) works "
          "without them; testing (cell 6) and the assistant (cell 7) need them.")

probe = "import torch; print(torch.__version__, torch.cuda.is_available())"
_rc, _out, _err = capture([sys.executable, "-c", probe])
if _rc != 0 or (GPU and "True" not in _out):
    idx = "https://download.pytorch.org/whl/cu128" if GPU else "https://download.pytorch.org/whl/cpu"
    # a CPU-only PyTorch (version "...+cpu") already installed has the same version number as the CUDA build, so pip
    # would keep it: force the CUDA build when this PC has an NVIDIA GPU
    run([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "--index-url", idx]
        + (["--force-reinstall", "--no-deps"] if GPU and "+cpu" in _out else []))
run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--progress-bar", "off",
     "rustbpe", "tiktoken", "filelock", "jsonschema", "wandb", "psutil", "pyarrow", "numpy", "requests", "gdown>=6",
     "pyyaml"])
importlib.invalidate_caches()
_rc, _out, _err = capture([sys.executable, "-c", probe])
if _rc != 0:
    err = _err[-800:]
    if "WinError 126" in err or "WinError 193" in err:
        raise SystemExit("PyTorch cannot load its DLLs: this PC is missing the Microsoft Visual C++ Redistributable. "
                         "Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe (choose Repair if it is "
                         "already there), restart the PC, then run cells 1 and 2 again.")
    if "4551" in err:
        raise SystemExit("Windows Smart App Control blocks PyTorch: Windows Security > App & browser control > "
                         "Smart App Control settings > Off, then run cells 1 and 2 again.")
    raise SystemExit("PyTorch does not load:\\n" + err)
print("PyTorch", _out.strip())
CUDA = GPU and _out.strip().endswith("True")
if MODE in ("long", "curriculum", "big"):
    raise SystemExit(f'MODE "{MODE}" was removed: it trained the old plain-prompt format and would undo the v4 model. '
                     'Use "refine" (GPU), "topup" (CPU) or "full" (GPU).')
if MODE not in ("auto", "refine", "full", "topup", "install", "rl", "pretrain"):
    raise SystemExit(f'MODE "{MODE}" is not known: use "auto", "refine", "full", "topup", "install", "rl" or "pretrain"')
if BASE not in ("current", "v6"):
    raise SystemExit(f'BASE "{BASE}" is not known: use "current" or "v6"')
MODE_NOW = MODE if MODE != "auto" or not CUDA else "auto"      # auto with a GPU: cell 3 looks on Drive, then decides
if MODE_NOW == "auto":
    print('MODE "auto" with an NVIDIA GPU: cell 3 picks "refine" when Drive (or this PC) has a trained model, else "full"')
elif MODE == "auto":
    MODE_NOW = "topup"
    print('MODE "auto" -> "topup": no NVIDIA GPU (a CPU top-up of the newest trained model)')
if REFINE_FROM not in ("auto", "drive", "pc"):
    raise SystemExit(f'REFINE_FROM "{REFINE_FROM}" is not known: use "auto", "drive" or "pc"')
if MODE_NOW == "rl" and not CUDA:
    raise SystemExit('MODE "rl" needs an NVIDIA GPU (each step writes 64 answers: about an hour per step on a CPU).')
if MODE_NOW == "refine" and not CUDA:
    raise SystemExit('MODE "refine" is sized for an NVIDIA GPU (1500 steps: about a day on a CPU). Use MODE = "topup".')
if MODE_NOW == "pretrain" and not CUDA:
    raise SystemExit('MODE "pretrain" needs an NVIDIA GPU (about 11-14 h on an RTX 4060; months on a CPU).')
if MODE_NOW == "full" and not CUDA:
    raise SystemExit('MODE "full" needs an NVIDIA GPU (12000 steps; weeks on a CPU). Use MODE = "topup" on this PC.')
try:
    import psutil
    RAM_GB = psutil.virtual_memory().total / 2**30
except Exception:
    RAM_GB = 8.0
# CPU: one 4096-token row per pass needs about 5-6 GB for training; below 10 GB RAM train at 2048 (the model then
# serves at 2048 too). Same tokens per step either way (the rest is gradient accumulation).
SEQ = SFT_SEQ if (CUDA or RAM_GB >= 10) else 2048
print("mode:", MODE_NOW, "| window:", SEQ, f"| RAM {RAM_GB:.0f} GB")

# the desktop launcher and the nightly top-up use the same Python as this notebook
bat = ROOT / "assistant" / "start_assistant.bat"
if bat.exists():
    import re as _re
    txt = bat.read_text(encoding="utf-8", errors="replace")
    with open(bat, "w", encoding="utf-8", newline="\\r\\n") as _f:     # cmd.exe wants Windows line ends
        _f.write(_re.sub(r"(?m)^set PYTHON=.*$", lambda m: "set PYTHON=" + sys.executable, txt))
''')

code('''
#@title 3. Download the model and the training data from Google Drive
# Drive sft_render/ (already rendered by the Colab run: saves about 50 minutes) is downloaded when it changed since
# the last time; without it, sft/ (+ sft_clean/) is downloaded and rendered here with the Colab cell 6/6d settings.
# A top-up starts from Drive's newest trained model (chatsft_checkpoints, else the newest chatsft_checkpoints_before_*).
# Files named "... (1).jsonl" on Drive are second uploads: they are reported and ignored.
import importlib
importlib.invalidate_caches()
sys.path.insert(0, str(LOCAL))
import data_pc
importlib.reload(data_pc)
assert DRIVE_FOLDER_ID, "set DRIVE_FOLDER_ID in cell 1"
DRIVE = data_pc.GdownDrive(DRIVE_FOLDER_ID)
print(len(DRIVE.files), "files on Drive")
if MODE_NOW == "auto":
    MODE_NOW, _why = data_pc.choose_mode("auto", CUDA, DRIVE.files, ROOT)
    print(f'MODE "auto" -> "{MODE_NOW}": {_why}')
GOT = data_pc.sync(DRIVE, ROOT, "install" if MODE_NOW == "rl" else MODE_NOW, refresh=REFRESH_DATA, seq=SEQ,  # rl: Drive's model + data
                   base=BASE, pretrain_data=PRETRAIN_DATA)
DATA = Path(GOT["data"])
# the assistant needs a tokenizer even before the first install (the one of the model it will get)
if not (ROOT / "nanochat_base" / "tokenizer" / "tokenizer.pkl").exists() and MODE_NOW != "pretrain":
    src = ROOT / "train_data" / (("base_v6" if BASE == "v6" else "base") if MODE_NOW == "full" else "drive_model") / "tokenizer"
    shutil.copytree(src, ROOT / "nanochat_base" / "tokenizer", dirs_exist_ok=True)
print("downloads done")
''')

code('''
#@title 4. Check everything is in place (and size the top-up)
TRAIN = [sys.executable, "-u", str(LOCAL / "train_pc.py"), "--rpa", str(ROOT), "--mode", MODE_NOW, "--data", str(DATA),
         "--seq", str(SEQ)]
RUN = "full_v6" if MODE_NOW == "full" and BASE == "v6" else MODE_NOW      # train_pc.run_name: the run's state folder
if MODE_NOW in ("refine", "topup"):
    TRAIN += ["--start-from", REFINE_FROM]   # its weights are checked first: a broken (nan) model is refused
def _refine_states():
    """(refine tested, RL after it: None / "going" / "done") for REFINE_THEN_RL."""
    _rs, _ls = ROOT / "local_run" / "refine" / "state.json", ROOT / "local_run" / "rl" / "state.json"
    _r = json.loads(_rs.read_text(encoding="utf-8")) if _rs.exists() else {}
    if not _r.get("finished"):
        return False, None
    if not _ls.exists() or _ls.stat().st_mtime < _rs.stat().st_mtime:      # an RL run from before this refine
        return True, None
    return True, "done" if json.loads(_ls.read_text(encoding="utf-8")).get("finished") else "going"
if MODE_NOW == "pretrain":     # the clean base model (training/local/pretrain_pc.py): data, tokenizer, pretrain, pack
    TRAIN = [sys.executable, "-u", str(LOCAL / "pretrain_pc.py"), "--rpa", str(ROOT), "--profile", PRETRAIN_PROFILE,
             "--data-from", PRETRAIN_DATA, "--render", str(DATA)]
    if (GOT.get("pretrain") or {}).get("tokenizer"):
        TRAIN += ["--tokenizer", GOT["pretrain"]["tokenizer"]]   # Drive's v6 tokenizer (Colab made it): the same one
    if PRETRAIN_MEASURE > 0:
        TRAIN += ["--measure", str(PRETRAIN_MEASURE)]
elif MODE_NOW == "rl":         # RL (training/local/rl_pc.py, the Colab cell 9 recipe) on the newest trained model
    TRAIN = [sys.executable, "-u", str(LOCAL / "rl_pc.py"), "--rpa", str(ROOT), "--data", str(DATA),
             "--steps", str(RL_STEPS), "--start-from", REFINE_FROM]
elif MODE_NOW == "full":
    TRAIN += ["--steps", str(FULL_STEPS), "--chunks", str(FULL_CHUNKS), "--base", BASE]
elif MODE_NOW == "refine":     # continue the finished Colab model (train_pc.py --mode refine, state local_run/refine)
    TRAIN += ["--steps", str(REFINE_STEPS), "--chunks", str(REFINE_CHUNKS)]
elif MODE_NOW == "topup":
    steps = TOPUP_STEPS
    _st = ROOT / "local_run" / "topup" / "state.json"           # a round in progress keeps its own length
    _going = _st.exists() and not json.loads(_st.read_text(encoding="utf-8")).get("finished")
    if TOPUP_HOURS > 0 and not (_going and not RESTART_TRAINING):
        print(f"measuring this PC's speed (2 steps, about 5 minutes) to size a {TOPUP_HOURS} h top-up ...")
        out = subprocess.run(TRAIN + ["--measure", "2"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        import re as _re
        m = _re.search(r"MEASURED: ([\\d.]+) s per step.*?first step \\+ loading (\\d+) s", out.stdout)
        if m:
            # measured as the run trains (low priority, one core left free); every 50-step chunk also loads the model
            # and the data again, about what the measuring run took beyond its steps
            sps, total = float(m.group(1)), float(m.group(2))
            load = max(0.0, total - 2 * sps)
            steps = max(50, int(TOPUP_HOURS * 3600 / (sps + load / 50)) // 50 * 50)
            print(f"  {sps:.0f} s per step + {load / 60:.0f} min loading per 50 steps -> {steps} steps for about "
                  f"{TOPUP_HOURS} h")
        else:
            print(out.stdout[-1500:], out.stderr[-1500:])
            print("  could not measure; using TOPUP_STEPS =", TOPUP_STEPS)
    TRAIN += ["--steps", str(steps)]
if RESTART_TRAINING:
    TRAIN += ["--restart"]
if MODE_NOW == "install":
    print("MODE install: no training; cell 6 tests Drive's newest model against this PC's")
else:
    rc = stream(TRAIN + ["--check"])
    print("\\nOK: run cell 5" if rc == 0 else "\\nfix the lines marked ! above, then run this cell again")
''')

code('''
#@title 5. Train (resumable: after any stop run cells 1-5 again and it continues)
# Keep the PC on and plugged in (it is kept awake while this runs). The installed model is not touched here.
if MODE_NOW == "install":
    print("MODE install: nothing to train")
elif MODE_NOW == "refine" and REFINE_THEN_RL and not RESTART_TRAINING and _refine_states()[1] == "going":
    print("the refine is trained and tested; its RL run is in progress: run cell 6 to continue it (nothing to do here)")
else:
    rc = stream(TRAIN)
    print("\\nexit code", rc, "| log:", ROOT / "local_run" / RUN / ("pretrain.log" if MODE_NOW == "pretrain" else "train.log"))
    if rc != 0:
        raise SystemExit("training stopped (the lines above say why; a STOPPED line says what to do and send); run "
                         "this cell again to continue from the last saved chunk")
''')

code('''
#@title 6. Test (as Colab cell 8) and install the new model only if it is better
if MODE_NOW == "pretrain":
    raise SystemExit("MODE pretrain makes a base model (it continues text, it does not answer yet): nothing to test "
                     "here. Next: cell 1 MODE = 'full' and BASE = 'v6', then cells 1-6 (the OOA training on it; cell 6 "
                     "installs it only if it beats the installed model).")
if not setup_pc.platform_ok(ROOT):
    raise SystemExit(setup_pc.PLATFORM_HELP.format(p=ROOT / "platform", rpa=ROOT) + ", then run this cell again")
if MODE_NOW == "install":
    _d8 = ROOT / "train_data" / "drive_model" / "chatsft_checkpoints" / "d8"
    _origin = (ROOT / "train_data" / "drive_model" / ".drive_id")
    EVAL = [sys.executable, "-u", str(LOCAL / "eval_pc.py"), "--rpa", str(ROOT), "--new", str(_d8), "--data", str(DATA),
            "--info", json.dumps({"mode": "install",
                                 "drive_origin": _origin.read_text(encoding="utf-8").strip() if _origin.exists() else None})]
else:
    _st = ROOT / "local_run" / RUN / "state.json"
    _res = json.loads(_st.read_text(encoding="utf-8")).get("result") if _st.exists() else None
    assert _res, "no finished training run: run cell 5 first"
    EVAL = [sys.executable, "-u", str(LOCAL / "eval_pc.py"), "--rpa", str(ROOT), "--new", _res, "--data", str(DATA),
            "--run-state", str(_st)]
_EV_OPTS = ["--per-set", str(0 if CUDA else EVAL_PER_SET), "--minutes", str(EVAL_MINUTES), "--install"]
_ref_tested, _rl_after = _refine_states() if MODE_NOW == "refine" and REFINE_THEN_RL else (False, None)
if _ref_tested and not RESTART_TRAINING:
    # this cell runs again to continue a stopped RL run: the refine was tested already, do not test it again
    print("the refine was tested already (see the history below): going on with RL")
    rc = 0
else:
    _rl_after = None
    rc = stream(EVAL + _EV_OPTS)
print("\\nexit code", rc, "| history:", ROOT / "nanochat_base" / "topup_history.jsonl")
if _rl_after == "done":
    print("the refine and its RL run are both trained and tested: nothing left to do here. Cell 7 starts the "
          "assistant (running cells 5 and 6 again would start a new refine).")
elif MODE_NOW == "refine" and rc == 0 and REFINE_THEN_RL:
    # RL starts from the newest installed model (the refined one when it was installed above, else the Colab model).
    # REFINE_FROM = "pc": always this PC's installed model (a refine from it keeps the PC's old Drive origin, so
    # "auto" would go back to Drive's model, the one refused as broken); otherwise "auto" (after a refine from Drive's
    # model was installed, "auto" picks it).
    _rl_from = "pc" if REFINE_FROM == "pc" else "auto"
    print(f"\\nREFINE_THEN_RL: reinforcement learning on the newest model (start {_rl_from}; about 3 h per 100 "
          "steps) ...")
    _rl_rc = stream([sys.executable, "-u", str(LOCAL / "rl_pc.py"), "--rpa", str(ROOT), "--data", str(DATA),
                     "--steps", str(RL_STEPS), "--start-from", _rl_from] + (["--restart"] if RESTART_TRAINING else []))
    if _rl_rc == 0:
        rc = stream([sys.executable, "-u", str(LOCAL / "eval_pc.py"), "--rpa", str(ROOT), "--data", str(DATA),
                     "--run-state", str(ROOT / "local_run" / "rl" / "state.json")] + _EV_OPTS)
    elif _rl_rc == 3:       # rl_pc.REFUSED: nothing started
        print("RL did NOT start: the lines marked ! above say why. Running this cell again will not change that: fix "
              "what they name first (a setting in cell 1: change it, run cell 1, then this cell again).")
    else:
        print("RL stopped: run this cell again to continue it (saved every 25 steps; cells 1-4 first after a restart "
              "are fine, cell 5 then skips the finished refine)")
elif MODE_NOW == "refine" and rc == 0:
    print('\\nNext (optional): MODE = "rl" in cell 1, then run cells 1-6 again: RL continues the newest installed '
          "model. Then cell 7 starts the assistant.")
''')

code('''
#@title 7. Start the assistant (opens in your browser)
import urllib.request, webbrowser
if not setup_pc.platform_ok(ROOT):
    raise SystemExit(setup_pc.PLATFORM_HELP.format(p=ROOT / "platform", rpa=ROOT) + ", then run this cell again")
if not setup_pc.nanochat_ok(ROOT / "nanochat"):
    raise SystemExit("nanochat is missing: run cell 2 again")
cat = ROOT / "sft" / "catalog.jsonl"             # the installed model's catalog (examples.jsonl beside it)
cat = cat if cat.exists() else ROOT / "data" / "catalog" / "catalog.jsonl"
(ROOT / "assistant" / "assistant_data").mkdir(parents=True, exist_ok=True)
APP_LOG = ROOT / "assistant" / "assistant_data" / "assistant.log"      # its messages (a notebook does not show them)
_app_log = open(APP_LOG, "w", encoding="utf-8")
APP = subprocess.Popen([sys.executable, str(ROOT / "assistant" / "app.py"), "--model-dir", str(ROOT / "nanochat_base"),
                        "--nanochat", str(ROOT / "nanochat"), "--platform", str(ROOT / "platform"), "--catalog", str(cat),
                        "--data-dir", str(ROOT / "assistant" / "assistant_data"), "--port", "8765"],
                       cwd=str(ROOT / "assistant"), stdout=_app_log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
for _ in range(150):
    time.sleep(2)
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8765/api/status",
                                                      headers={"X-OOA-Assistant": "1"}), timeout=5)
        print("OOA Assistant is running: http://127.0.0.1:8765  (run cell 8 to stop it)")
        webbrowser.open("http://127.0.0.1:8765")
        break
    except Exception:
        if APP.poll() is not None:
            _app_log.close()
            print(APP_LOG.read_text(encoding="utf-8", errors="replace")[-3000:])
            raise SystemExit("the assistant stopped while starting; its messages are above (full log: "
                             + str(APP_LOG) + ")")
else:
    print("the assistant is still starting; its messages so far:", APP_LOG)
''')

code('''
#@title 8. Stop the assistant
if os.name == "nt":
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(APP.pid)], capture_output=True)
else:
    APP.terminate()
APP.wait()
print("stopped")
''')

code('''
#@title 9. (Windows, optional) Top-up every night at 01:30 (continues the run, then tests and installs if better)
NIGHTLY = ROOT / "training" / "local" / "topup_nightly.bat"
if os.name != "nt":
    raise SystemExit("this cell is for Windows (Task Scheduler); on Linux use cron with train_pc.py and eval_pc.py")
_rc = stream(["schtasks", "/create", "/tn", "OOA nightly top-up", "/tr", f'"{NIGHTLY}"', "/sc", "daily", "/st", "01:30", "/f"])
stream(["powershell", "-NoProfile", "-Command",
        "Set-ScheduledTask -TaskName 'OOA nightly top-up' -Settings (New-ScheduledTaskSettingsSet "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 8))"])
if _rc == 0:
    print("scheduled: every night at 01:30 (results in", NIGHTLY.parent / "topup_nightly.log", ")")
else:
    print("could not schedule it (see the messages above)")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT)
