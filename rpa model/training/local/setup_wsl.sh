#!/usr/bin/env bash
# One-time setup for local training, inside WSL (Ubuntu) or Linux with an NVIDIA GPU.
#   tr -d '\r' < "/mnt/c/ooa/rpa model/training/local/setup_wsl.sh" | bash
# (tr strips Windows line endings, in case the file passed through a Windows tool.)
# Creates the Python environment ~/ooa-train-venv (about 6 GB with PyTorch). Safe to run again.
set -euo pipefail
VENV="${VENV:-$HOME/ooa-train-venv}"

echo "== 1/4 GPU check"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi does not work here. In WSL: install/update the NVIDIA driver in WINDOWS (not inside Ubuntu),"
  echo "restart Windows, open Ubuntu again and rerun this script."
  exit 1
fi

echo "== 2/4 system packages (asks for your Ubuntu password; may wait for Ubuntu's own first updates)"
sudo apt-get -o DPkg::Lock::Timeout=900 update -y
sudo apt-get -o DPkg::Lock::Timeout=900 install -y python3 python3-venv python3-dev build-essential git

echo "== 3/4 Python environment in $VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# the PyTorch version nanochat (commit 92d63d4) was written for; the Linux wheel includes CUDA
"$VENV/bin/pip" install "torch==2.9.1"
"$VENV/bin/pip" install rustbpe tiktoken kernels filelock pyarrow psutil requests wandb numpy jsonschema

echo "== 4/4 GPU + compiler test (about a minute)"
"$VENV/bin/python" - <<'EOF'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the GPU. Update the NVIDIA driver in Windows (570 or newer) and restart.")
name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability()
print("PyTorch", torch.__version__, "|", name, "| compute capability", f"{cap[0]}.{cap[1]}")
if cap < (7, 5):
    raise SystemExit(f"{name} is too old for the training code (needs an RTX 20xx / GTX 16xx or newer).")
try:
    f = torch.compile(lambda x: (x * 2).sin().sum())
    f(torch.randn(1024, device="cuda"))
    torch.cuda.synchronize()
except Exception as ex:
    raise SystemExit(f"torch.compile test failed: {ex}\nUpdate the NVIDIA driver in Windows (570 or newer), restart, "
                     "and run this setup again.")
print("GPU and compiler test: OK")
EOF
echo
echo "Setup done. Next (README.md step 5):"
echo "  ~/ooa-train-venv/bin/python \"/mnt/c/ooa/rpa model/training/local/train_local.py\" --work /mnt/c/ooa_train --check"
