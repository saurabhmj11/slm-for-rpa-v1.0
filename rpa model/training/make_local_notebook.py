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
| `"full"` (default with an NVIDIA GPU) | **the Colab run**: 8000 steps at the 4096 window from the pretrained **base** model | estimated 2-3 h on an RTX 3060/4060 (the Colab T4 takes about 2.5 h); a CPU would need weeks, so it is not offered there |
| `"topup"` (default without a GPU) | **continues the newest trained model** (Drive's, or this PC's own when it already grew from it) for `TOPUP_STEPS` steps on the same data, mostly the beginner and long business-process rows | 300 steps: about 5.5-9 h on 4 CPU cores (measured on a 4-core test machine at the 4096 window: 67 s per step at low priority, 55 s with all cores; slower PCs up to about 100 s) |
| `"install"` | no training: tests Drive's newest model against this PC's and installs it if it is better | the test only (cell 6) |

Training is saved in chunks (top-up: every 50 steps; full: one run like Colab, or `FULL_CHUNKS` pieces): after a stop,
restart or crash run cells 1-5 again and it continues from the last saved chunk. On a CPU it runs at low priority with
one core left free, so the PC stays usable.

Cell 6 scores the new model and the installed one through the full assistant (catalog lock, repair, verify,
self-correction), on sets 1-3 and the beginner set, as Colab cell 8 does; on a CPU a sample (`EVAL_PER_SET` requests
per set, at most `EVAL_MINUTES` per model; a request takes 2-70 s on 4 cores, about 15-25 minutes per model). The new
model is **installed only if it is better** (the old one is kept in `nanochat_base/old_model_<date>`), together with
its `catalog.jsonl` and `examples.jsonl` in `rpa model/sft/`, so the assistant gives it the context + plan prompts it
was trained on.

Before the first run: unzip `ooa_assistant.zip`, open this notebook from its `rpa model` folder in VS Code (Python +
Jupyter extensions) or Jupyter, and pick a Python 3.10+ kernel. Internet is needed for cells 2 and 3. The Drive folder
`ooa-nanochat` must be shared as "anyone with the link"; put its id (from the link) in cell 1.
""")

code(f'''
#@title 1. Settings
DRIVE_FOLDER_ID = "{FOLDER_ID}"     # Google Drive folder ooa-nanochat (shared as "anyone with the link")
MODE = "auto"              # "auto": "full" with an NVIDIA GPU, "topup" without; or "full", "topup", "install"
TOPUP_STEPS = 300          # top-up length in steps of 32,768 tokens (4 CPU cores: about 5.5-9 h)
TOPUP_HOURS = 0            # > 0: measure this PC's speed first (about 5 min) and size the top-up to this many hours
FULL_STEPS = 8000          # the Colab run (SFT_ITERS 32000 micro-batches = 8000 steps)
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
    here = Path.cwd()
    starts += [here, *here.parents]
    home = Path.home()
    starts += [Path("C:/OOA"), home / "Downloads", home / "Desktop", home / "OneDrive" / "Desktop", home / "Documents", here]
    seen = set()
    for s0 in starts:
        for p in [s0, s0 / "rpa model"]:
            if p in seen or not p.exists():
                continue
            seen.add(p)
            if _ok(p):
                return p
    for s0 in starts[len(starts) - 6:]:          # one more level down, e.g. Downloads/ooa_assistant/rpa model
        if s0.exists():
            for p in list(s0.glob("*/rpa model")) + list(s0.glob("*/*/rpa model")):
                if _ok(p):
                    return p
    print("Looked in:", *[str(p) for p in seen], sep=chr(10) + "  ")
    raise SystemExit("Could not find the 'rpa model' folder of the NEW ooa_assistant.zip (it must contain "
                     "assistant/app.py and training/local/train_pc.py). Unzip the new zip to C:/OOA, "
                     "or put its path in RPA_FOLDER above, then run this cell again.")

ROOT = _find()
LOCAL = ROOT / "training" / "local"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
print("assistant folder:", ROOT)
print("Python:", sys.executable, sys.version.split()[0])
''')

code('''
#@title 2. Check this PC and install what is needed (first time: several minutes)
def run(cmd, check=True):
    print(">", " ".join(str(c) for c in cmd)[:200])
    r = subprocess.run([str(c) for c in cmd])
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (exit {r.returncode}); see the messages above")
    return r.returncode

def stream(cmd):
    """Run a step and show its output as it happens; -> exit code. A stopped cell ends the step with everything it
    started (Windows: the whole process tree, since terminating a process there does not end its children)."""
    print(">", " ".join(str(c) for c in cmd)[:200])
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONUNBUFFERED="1"))
    try:
        for line in p.stdout:
            print(line, end="")
    finally:
        if p.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            else:
                p.terminate()
    return p.wait()

GPU, GPU_MEM = False, 0
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    if q.returncode == 0 and q.stdout.strip():
        name, mem = q.stdout.strip().splitlines()[0].rsplit(",", 1)
        GPU, GPU_MEM = True, int(float(mem)) // 1024
        print(f"NVIDIA GPU: {name.strip()} ({GPU_MEM} GB)")
except FileNotFoundError:
    pass
if not GPU:
    print("no NVIDIA GPU: training will run on the CPU |", os.cpu_count(), "CPU cores")

if os.name == "nt":                     # Smart App Control blocks PyTorch's files (WinError 4551)
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\\CurrentControlSet\\Control\\CI\\Policy")
        if winreg.QueryValueEx(k, "VerifiedAndReputablePolicyState")[0] == 1:
            raise SystemExit("Windows Smart App Control is ON and will block PyTorch. Turn it off: Windows Security > "
                             "App & browser control > Smart App Control settings > Off, then run this cell again.")
    except OSError:
        pass

probe = "import torch; print(torch.__version__, torch.cuda.is_available())"
t = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
if t.returncode != 0 or (GPU and "True" not in t.stdout):
    idx = "https://download.pytorch.org/whl/cu128" if GPU else "https://download.pytorch.org/whl/cpu"
    run([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "--index-url", idx])
run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
     "rustbpe", "tiktoken", "filelock", "jsonschema", "wandb", "psutil", "pyarrow", "numpy", "requests", "gdown"])
t = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
if t.returncode != 0:
    err = t.stderr[-800:]
    if "WinError 126" in err or "WinError 193" in err:
        raise SystemExit("PyTorch cannot load its DLLs: this PC is missing the Microsoft Visual C++ Redistributable. "
                         "Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe (choose Repair if it is "
                         "already there), restart the PC, then run cells 1 and 2 again.")
    if "4551" in err:
        raise SystemExit("Windows Smart App Control blocks PyTorch: Windows Security > App & browser control > "
                         "Smart App Control settings > Off, then run cells 1 and 2 again.")
    raise SystemExit("PyTorch does not load:\\n" + err)
print("PyTorch", t.stdout.strip())
CUDA = GPU and t.stdout.strip().endswith("True")
MODE_NOW = MODE if MODE in ("full", "topup", "install") else ("full" if CUDA else "topup")
if MODE in ("long", "refine", "curriculum", "big"):
    raise SystemExit(f'MODE "{MODE}" was removed: it trained the old plain-prompt format and would undo the v4 model. '
                     'Use "topup" (CPU) or "full" (GPU).')
if MODE_NOW == "full" and not CUDA:
    raise SystemExit('MODE "full" needs an NVIDIA GPU (8000 steps; weeks on a CPU). Use MODE = "topup" on this PC.')
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
    bat.write_text(_re.sub(r"(?m)^set PYTHON=.*$", lambda m: "set PYTHON=" + sys.executable, txt), encoding="utf-8")
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
GOT = data_pc.sync(DRIVE, ROOT, MODE_NOW, refresh=REFRESH_DATA, seq=SEQ)
DATA = Path(GOT["data"])
# the assistant needs a tokenizer even before the first install
if not (ROOT / "nanochat_base" / "tokenizer" / "tokenizer.pkl").exists():
    src = ROOT / "train_data" / ("base" if MODE_NOW == "full" else "drive_model") / "tokenizer"
    shutil.copytree(src, ROOT / "nanochat_base" / "tokenizer", dirs_exist_ok=True)
print("downloads done")
''')

code('''
#@title 4. Check everything is in place (and size the top-up)
TRAIN = [sys.executable, "-u", str(LOCAL / "train_pc.py"), "--rpa", str(ROOT), "--mode", MODE_NOW, "--data", str(DATA),
         "--seq", str(SEQ)]
if MODE_NOW == "full":
    TRAIN += ["--steps", str(FULL_STEPS), "--chunks", str(FULL_CHUNKS)]
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
    r = subprocess.run(TRAIN + ["--check"])
    print("\\nOK: run cell 5" if r.returncode == 0 else "\\nfix the lines marked ! above, then run this cell again")
''')

code('''
#@title 5. Train (resumable: after any stop run cells 1-5 again and it continues)
# Keep the PC on and plugged in (it is kept awake while this runs). The installed model is not touched here.
if MODE_NOW == "install":
    print("MODE install: nothing to train")
else:
    rc = stream(TRAIN)
    print("\\nexit code", rc, "| log:", ROOT / "local_run" / MODE_NOW / "train.log")
    if rc != 0:
        raise SystemExit("training stopped; run this cell again to continue from the last saved chunk")
''')

code('''
#@title 6. Test (as Colab cell 8) and install the new model only if it is better
if MODE_NOW == "install":
    _d8 = ROOT / "train_data" / "drive_model" / "chatsft_checkpoints" / "d8"
    _origin = (ROOT / "train_data" / "drive_model" / ".drive_id")
    EVAL = [sys.executable, "-u", str(LOCAL / "eval_pc.py"), "--rpa", str(ROOT), "--new", str(_d8), "--data", str(DATA),
            "--info", json.dumps({"mode": "install", "drive_origin": _origin.read_text().strip() if _origin.exists() else None})]
else:
    _st = ROOT / "local_run" / MODE_NOW / "state.json"
    _res = json.loads(_st.read_text(encoding="utf-8")).get("result") if _st.exists() else None
    assert _res, "no finished training run: run cell 5 first"
    EVAL = [sys.executable, "-u", str(LOCAL / "eval_pc.py"), "--rpa", str(ROOT), "--new", _res, "--data", str(DATA),
            "--run-state", str(_st)]
EVAL += ["--per-set", str(0 if CUDA else EVAL_PER_SET), "--minutes", str(EVAL_MINUTES), "--install"]
rc = stream(EVAL)
print("\\nexit code", rc, "| history:", ROOT / "nanochat_base" / "topup_history.jsonl")
''')

code('''
#@title 7. Start the assistant (opens in your browser)
import urllib.request, webbrowser
cat = ROOT / "sft" / "catalog.jsonl"             # the installed model's catalog (examples.jsonl beside it)
cat = cat if cat.exists() else ROOT / "data" / "catalog" / "catalog.jsonl"
APP = subprocess.Popen([sys.executable, str(ROOT / "assistant" / "app.py"), "--model-dir", str(ROOT / "nanochat_base"),
                        "--nanochat", str(ROOT / "nanochat"), "--platform", str(ROOT / "platform"), "--catalog", str(cat),
                        "--data-dir", str(ROOT / "assistant" / "assistant_data"), "--port", "8765"],
                       cwd=str(ROOT / "assistant"))
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
            raise SystemExit("the assistant stopped while starting; see the messages above")
''')

code('''
#@title 8. Stop the assistant
APP.terminate(); print("stopped")
''')

code('''
#@title 9. (Windows, optional) Top-up every night at 01:30 (continues the run, then tests and installs if better)
NIGHTLY = ROOT / "training" / "local" / "topup_nightly.bat"
subprocess.run(["schtasks", "/create", "/tn", "OOA nightly top-up", "/tr", f'"{NIGHTLY}"', "/sc", "daily", "/st", "01:30", "/f"])
subprocess.run(["powershell", "-NoProfile", "-Command",
                "Set-ScheduledTask -TaskName 'OOA nightly top-up' -Settings (New-ScheduledTaskSettingsSet "
                "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 8))"])
print("scheduled: every night at 01:30 (results in", NIGHTLY.parent / "topup_nightly.log", ")")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT)
