"""Generate ooa_local_pc.ipynb: train and run the OOA model on any Windows/Linux PC (GPU or CPU), no Colab.
Run:  python make_local_notebook.py [DRIVE_FOLDER_ID]   (writes "rpa model/ooa_local_pc.ipynb")
The notebook sits in the "rpa model" folder of ooa_assistant.zip and drives training/local/*.py.
"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
FOLDER_ID = sys.argv[1] if len(sys.argv) > 1 else ""
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})

md("""
# OOA model on this PC: download, train, test, run

Run the cells from top to bottom (Shift+Enter). Works on Windows and Linux, with or without an NVIDIA graphics card:

| This PC has | Cell 5 does | Time |
|---|---|---|
| NVIDIA RTX 20xx / GTX 16xx or newer | **full training** from the pretrained base model (same recipe as Colab) | about 1-2 h |
| no graphics card | **top-up**: a short round from the current model on your feedback + training data | about 2-2.5 h on 4 cores |
| no graphics card, `MODE = "long"` in cell 1 | **long run**: the full Colab-size training in 10 saved chunks, at low priority so the PC stays usable; resumes after a restart | about 1-1.5 days on 4 cores |

Either way the new model is scored on the same 20 everyday requests and **installed only if it is better** than the
current one (the old one is kept in `nanochat_base/old_model_<date>`). Cell 6 starts the assistant in your browser.

Before the first run: unzip `ooa_assistant.zip`, open this notebook from its `rpa model` folder in VS Code (Python +
Jupyter extensions) or Jupyter, and pick a Python 3.10+ kernel. Internet is needed for cells 2 and 3.
""")

code(f'''
#@title 1. Settings
DRIVE_FOLDER_ID = "{FOLDER_ID}"     # Google Drive folder ooa-nanochat (shared as "anyone with the link")
MODE = "auto"              # "auto": "full" with an NVIDIA GPU, "topup" without. Or choose:
                           #   "long"  : Colab-size training on a CPU, day and night, in 10 saved chunks (resumable)
                           #   "full"  : Colab-size training in one go (GPU)    "topup": short round from the current model
TOPUP_STEPS = 200          # top-up length (CPU: about 2-2.5 h on 4 cores)
FULL_STEPS = 2500          # full training length (same as Colab)
REPLACE_LOCAL_MODEL = False   # True: replace this PC's model with the newest one from Drive (the old one is kept)
REFRESH_DATA = False       # True: download the training data again even if it is already here

RPA_FOLDER = r""           # optional: the 'rpa model' folder, e.g. r"C:/OOA/rpa model" (empty = search for it)

import os, sys, json, shutil, subprocess, time
from pathlib import Path

def _ok(p):
    return (p / "assistant" / "app.py").exists() and (p / "training" / "local" / "topup_local.py").exists()

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
                     "assistant/app.py and training/local/topup_local.py). Unzip the new zip to C:/OOA, "
                     "or put its path in RPA_FOLDER above, then run this cell again.")

ROOT = _find()
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
print("assistant folder:", ROOT)
print("Python:", sys.executable, sys.version.split()[0])
''')

code('''
#@title 2. Check this PC and install what is needed (first time: several minutes)
def run(cmd, check=True):
    print(">", " ".join(str(c) for c in cmd)[:160])
    r = subprocess.run([str(c) for c in cmd])
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (exit {r.returncode}); see the messages above")

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
MODE_NOW = MODE if MODE in ("full", "topup", "long") else ("full" if CUDA else "topup")
DEVICE_BATCH = (2 if MODE_NOW == "long" else 4) if not CUDA else (4 if GPU_MEM >= 10 else 2 if GPU_MEM >= 6 else 1)
print("training mode:", MODE_NOW, "| device batch:", DEVICE_BATCH)

# the desktop launcher and the nightly top-up use the same Python as this notebook
bat = ROOT / "assistant" / "start_assistant.bat"
if bat.exists():
    import re as _re
    txt = bat.read_text(encoding="utf-8", errors="replace")
    bat.write_text(_re.sub(r"(?m)^set PYTHON=.*$", lambda m: "set PYTHON=" + sys.executable, txt), encoding="utf-8")
''')

code('''
#@title 3. Download the model and training data from Google Drive (about 0.5 GB; 1.5 GB more for full training)
import importlib, re
importlib.invalidate_caches()
import gdown
assert DRIVE_FOLDER_ID, "set DRIVE_FOLDER_ID in cell 1"
print("listing the Drive folder (a minute) ...")
listing = gdown.download_folder(id=DRIVE_FOLDER_ID, skip_download=True, quiet=True)
FILES = {}
for f in listing:
    FILES.setdefault(str(f.path).replace("\\\\", "/"), []).append(f.id)
print(len(FILES), "files on Drive")

def fetch(drive_path, dest, refresh=False):
    """Download one Drive file; if Drive holds two with that name, keep the one with more lines (the fuller one)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    ids = FILES.get(drive_path)
    if not ids:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    best, best_n = None, -1
    for i, fid in enumerate(ids):
        tmp = dest.with_name(dest.name + f".part{i}")
        gdown.download(id=fid, output=str(tmp), quiet=True)
        n = sum(1 for _ in open(tmp, "rb")) if tmp.suffix != ".pt" and ".pt." not in tmp.name else tmp.stat().st_size
        if n > best_n:
            if best is not None:
                best.unlink()
            best, best_n = tmp, n
        else:
            tmp.unlink()
    best.replace(dest)
    print("  got", drive_path)
    return dest

def newest_pair(folder, kind="model"):
    """Highest step N in a Drive folder that has model_N.pt and meta_N.json."""
    steps = []
    for p in FILES:
        m = re.fullmatch(re.escape(folder) + r"/model_(\\d+)\\.pt", p)
        if m and f"{folder}/meta_{m.group(1)}.json" in FILES:
            steps.append(m.group(1))
    return max(steps, key=int) if steps else None

base = ROOT / "nanochat_base"
for name in ("tokenizer.pkl", "token_bytes.pt"):
    assert fetch(f"nanochat_base/tokenizer/{name}", base / "tokenizer" / name), f"{name} not found on Drive"

# the chat model: the live one, else the newest kept copy (never the autoresearch mini-model)
folders = ["nanochat_base/chatsft_checkpoints/d8"] + sorted(
    {p.rsplit("/", 1)[0] for p in FILES if re.match(r"nanochat_base/chatsft_checkpoints_before_\\d+_\\d+/d8/", p)}, reverse=True)
chat = next(((f, s) for f in folders if (s := newest_pair(f))), None)
d8 = base / "chatsft_checkpoints" / "d8"
have = sorted(d8.glob("model_*.pt")) if d8.exists() else []
if chat and (not have or REPLACE_LOCAL_MODEL):
    if have:
        keep = base / f"old_model_{time.strftime('%Y%m%d_%H%M%S')}"
        keep.mkdir()
        for f in d8.iterdir():
            shutil.move(str(f), keep / f.name)
        print("previous model kept in", keep)
    folder, s = chat
    fetch(f"{folder}/model_{s}.pt", d8 / f"model_{s}.pt", refresh=True)
    fetch(f"{folder}/meta_{s}.json", d8 / f"meta_{s}.json", refresh=True)
    print("chat model:", folder, "step", s)
elif have:
    print("keeping this PC's model:", have[-1].name, "(set REPLACE_LOCAL_MODEL = True to take Drive's newest)")
else:
    raise SystemExit("no trained chat model found on Drive")

# training data
SFT = ["natural_train.jsonl", "core_train.jsonl", "chains_train.jsonl", "synth_train.jsonl", "synth_pattern.jsonl", "train_flows.jsonl",
       "train_catalog.jsonl", "val.jsonl", "catalog.jsonl", "self_improved.jsonl", "hard_fixed.jsonl"]
for name in SFT:
    fetch(f"sft/{name}", ROOT / "sft" / name, refresh=REFRESH_DATA)

# full training starts from the pretrained base model
if MODE_NOW in ("full", "long"):
    s = newest_pair("nanochat_base/base_checkpoints/d8")
    assert s, "no pretrained base model on Drive (nanochat_base/base_checkpoints/d8)"
    for name in (f"model_{s}.pt", f"meta_{s}.json", f"optim_{s}_rank0.pt"):
        fetch(f"nanochat_base/base_checkpoints/d8/{name}", base / "base_checkpoints" / "d8" / name)
    print("base model step", s)
print("downloads done")
''')

code('''
#@title 4. Check everything is in place
if MODE_NOW == "long":     # 10 saved chunks; run cell 5 again after any interruption to continue
    TOPUP = [sys.executable, "-u", str(ROOT / "training" / "local" / "train_long.py"), "--rpa", str(ROOT),
             "--chunks", "10", "--chunk-steps", str(max(1, FULL_STEPS // 10)), "--device-batch", str(DEVICE_BATCH)]
else:
    TOPUP = [sys.executable, "-u", str(ROOT / "training" / "local" / "topup_local.py"), "--rpa", str(ROOT), "--mode", MODE_NOW,
             "--steps", str(FULL_STEPS if MODE_NOW == "full" else TOPUP_STEPS), "--device-batch", str(DEVICE_BATCH)]
r = subprocess.run(TOPUP + ["--check"])
print("\\nOK: run cell 5" if r.returncode == 0 else "\\nfix the lines marked ! above, then run this cell again")
''')

code('''
#@title 5. Train, test, and install the new model only if it is better
# Keep the PC on and plugged in. The current model is only replaced at the very end, after scoring.
# MODE "long": each finished chunk is saved; after a stop, restart or crash just run this cell again to continue.
# It runs at low priority with one CPU core left free, so you can keep using the PC (slower, but smooth).
p = subprocess.Popen(TOPUP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                     env=dict(os.environ, PYTHONUNBUFFERED="1"))   # show progress as it happens
try:
    for line in p.stdout:
        print(line, end="")
finally:
    if p.poll() is None:
        p.terminate()
print("\\nexit code", p.wait(), "| full training log: the newest topup_*.log in", ROOT)
''')

code('''
#@title 6. Start the assistant (opens in your browser)
import urllib.request, webbrowser
cat = ROOT / "sft" / "catalog.jsonl"
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
        print("OOA Assistant is running: http://127.0.0.1:8765  (run cell 7 to stop it)")
        webbrowser.open("http://127.0.0.1:8765")
        break
    except Exception:
        if APP.poll() is not None:
            raise SystemExit("the assistant stopped while starting; see the messages above")
''')

code('''
#@title 7. Stop the assistant
APP.terminate(); print("stopped")
''')

code('''
#@title 8. (Windows, optional) Top-up every night at 01:30 with your feedback
NIGHTLY = ROOT / "training" / "local" / "topup_nightly.bat"
subprocess.run(["schtasks", "/create", "/tn", "OOA nightly top-up", "/tr", f'"{NIGHTLY}"', "/sc", "daily", "/st", "01:30", "/f"])
subprocess.run(["powershell", "-NoProfile", "-Command",
                "Set-ScheduledTask -TaskName 'OOA nightly top-up' -Settings (New-ScheduledTaskSettingsSet "
                "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 8))"])
print("scheduled: every night at 01:30 (results in", NIGHTLY.parent / "topup_nightly.log", ")")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = H.parent / "ooa_local_pc.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
