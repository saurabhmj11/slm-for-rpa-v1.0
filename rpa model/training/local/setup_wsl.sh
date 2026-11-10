#!/usr/bin/env bash
# One-time setup for local training, inside WSL (Ubuntu) or Linux with an NVIDIA GPU.
#   bash "/mnt/c/ooa/rpa model/training/local/setup_wsl.sh"
# Creates the Python environment ~/ooa-train-venv (about 6 GB with PyTorch). Safe to run again.
set -euo pipefail
VENV="${VENV:-$HOME/ooa-train-venv}"

echo "== 1/3 GPU check"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi does not work here. In WSL: install/update the NVIDIA driver in WINDOWS (not inside Ubuntu),"
  echo "restart Windows, open Ubuntu again and rerun this script."
  exit 1
fi

echo "== 2/3 system packages (asks for your Ubuntu password)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-dev build-essential git

echo "== 3/3 Python environment in $VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# the PyTorch version nanochat (commit 92d63d4) was written for; the Linux wheel includes CUDA
"$VENV/bin/pip" install "torch==2.9.1"
"$VENV/bin/pip" install rustbpe tiktoken kernels filelock pyarrow psutil requests wandb numpy jsonschema
"$VENV/bin/python" - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("PyTorch", torch.__version__, "| GPU visible:", ok, "|", torch.cuda.get_device_name(0) if ok else "-")
if not ok:
    raise SystemExit("PyTorch cannot see the GPU. Update the NVIDIA driver in Windows and restart.")
EOF
echo
echo "Setup done. Next (see README.md):"
echo "  $VENV/bin/python \"$(dirname "$(readlink -f "$0")")/train_local.py\" --work /mnt/c/ooa_train --check"
