"""Make ooa_nanochat_kaggle.ipynb from ooa_nanochat_colab.ipynb (run make_colab_notebook.py first).
Changes: no Google Drive (files live in /kaggle/working), /content -> /kaggle/working, internet check, restore of an
earlier model from a Kaggle dataset, and a final cell that packages the model for download."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
nb = json.loads((H / "ooa_nanochat_colab.ipynb").read_text(encoding="utf-8"))

SETUP = '''
#@title 2. Setup (Kaggle): GPU, internet, nanochat, dependencies
import os, sys, shutil, glob, urllib.request, torch
assert torch.cuda.is_available(), "No GPU. Right panel: Settings -> Accelerator -> GPU T4 x2 (or P100)"
cap = torch.cuda.get_device_capability()
print(torch.cuda.get_device_name(0), "compute capability", cap, "| GPUs:", torch.cuda.device_count(), "(this notebook uses one)")
try:
    urllib.request.urlopen("https://github.com", timeout=10)
except Exception:
    raise RuntimeError("No internet. Right panel: Settings -> Internet -> On (Kaggle asks for phone verification once).")
os.makedirs(DRIVE_DIR, exist_ok=True)
os.environ["NANOCHAT_BASE_DIR"] = DRIVE_DIR + "/nanochat_base"
# pre-Ampere GPUs (T4, P100) have no bf16: use fp16 with loss scaling. If you see NaN losses, set "float32" (slower).
os.environ["NANOCHAT_DTYPE"] = "bfloat16" if cap >= (8, 0) else "float16"
os.environ["OMP_NUM_THREADS"] = "1"
print("compute dtype:", os.environ["NANOCHAT_DTYPE"])

# restore an earlier model that you added as a Kaggle dataset (see the note at the top)
base = os.environ["NANOCHAT_BASE_DIR"]
for src in glob.glob("/kaggle/input/*/nanochat_base") + glob.glob("/kaggle/input/*/*/nanochat_base"):
    if not os.path.exists(base):
        shutil.copytree(src, base)
        print("restored model files from", src)

if not os.path.exists("/content/nanochat"):
    !git clone -q https://github.com/karpathy/nanochat /content/nanochat
!cd /content/nanochat && git checkout -q {NANOCHAT_COMMIT}
# Keep Kaggle's own torch; install only nanochat's other dependencies
!pip -q install rustbpe tiktoken kernels filelock pyarrow psutil requests wandb jsonschema
%cd /content/nanochat
'''

BUNDLE = '''
#@title 11. Package the trained model (download it from the notebook's Output tab after "Save Version")
import glob, os, re, shutil
base = os.environ["NANOCHAT_BASE_DIR"]
out = "/content/ooa_model_bundle"
shutil.rmtree(out, ignore_errors=True)
for src in ("base_checkpoints", "chatsft_checkpoints", "chatrl_checkpoints"):
    d = f"{base}/{src}/d{P['depth']}"
    steps = sorted(glob.glob(d + "/model_*.pt"))
    if not steps:
        continue
    tag = re.search(r"model_(\\d+)\\.pt", steps[-1]).group(1)
    dst = f"{out}/nanochat_base/{src}/d{P['depth']}"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(d + f"/*_{tag}*"):          # model, meta and optimizer files of the latest step
        shutil.copy(f, dst)
    print("packed", src, "step", tag)
shutil.copytree(f"{base}/tokenizer", f"{out}/nanochat_base/tokenizer")
shutil.make_archive("/content/ooa_model_bundle", "zip", out)
print("bundle:", round(os.path.getsize("/content/ooa_model_bundle.zip") / 1e6), "MB at /content/ooa_model_bundle.zip")
'''

INTRO = """
# OOA model with nanochat on **Kaggle**: pretrain, then train on our use case

Same pipeline as the Colab notebook. Set these in the right-hand panel first:
**Accelerator -> GPU T4 x2 (or P100)**, **Internet -> On**, and **Persistence** as you like.

Kaggle keeps files in `/kaggle/working` only for the session. To keep your work:
1. At the end run the last cell, then **Save Version** (Save & Run All, or Quick Save). Download `ooa_model_bundle.zip` from the Output tab.
2. To continue later, upload that zip as a **Kaggle Dataset** (or add this notebook's output as an input). Cell 2 finds the
   `nanochat_base` folder in `/kaggle/input` and restores it, so training resumes.

**If you already pretrained** (model files from Colab): put them in a dataset, then run cells 1, 2, then skip to cell 6.
Skip cells 3-5 (data download, tokenizer and pretraining), they are only for pretraining from zero.
Free Kaggle gives about 30 GPU hours a week; one session can run up to 12 hours.
"""

def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    cell["source"] = text.strip("\n").splitlines(True)

new = []
for c in nb["cells"]:
    s = src(c)
    if c["cell_type"] == "markdown" and s.lstrip().startswith("# OOA model with nanochat"):
        set_src(c, INTRO)
    elif c["cell_type"] == "code" and "#@title 2. Setup" in s:
        set_src(c, SETUP)
    else:
        s = s.replace("/content/drive/MyDrive/ooa-nanochat", "/kaggle/working/ooa-nanochat")
        s = s.replace("/content", "/kaggle/working")
        set_src(c, s) if c["cell_type"] == "code" else None
        if c["cell_type"] == "code":
            pass
    new.append(c)
# the setup/bundle cells are written with /content placeholders: apply the same path swap to them
for c in new:
    if c["cell_type"] == "code":
        set_src(c, src(c).replace("/content", "/kaggle/working"))
bundle_cell = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": []}
set_src(bundle_cell, BUNDLE.replace("/content", "/kaggle/working"))
new.append(bundle_cell)
nb["cells"] = new
nb["metadata"]["accelerator"] = "GPU"
text = json.dumps(nb, indent=1)
assert "google.colab" not in text and '"/content' not in text, "leftover Colab-only paths"
(H / "ooa_nanochat_kaggle.ipynb").write_text(text, encoding="utf-8")
print("wrote", H / "ooa_nanochat_kaggle.ipynb", "| cells:", len(new))
