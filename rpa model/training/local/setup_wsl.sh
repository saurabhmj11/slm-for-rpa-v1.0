#!/usr/bin/env bash
# One-time setup for local training, inside WSL (Ubuntu) or Linux with an NVIDIA GPU.
#   tr -d '\r' < "/mnt/c/ooa/rpa model/training/local/setup_wsl.sh" | bash
# (tr strips Windows line endings, in case the file passed through a Windows tool.)
# PC without an NVIDIA GPU (CPU only, for overnight top-ups):
#   tr -d '\r' < "/mnt/c/ooa/rpa model/training/local/setup_wsl.sh" | bash -s -- --cpu
# Creates the Python environment ~/ooa-train-venv (about 6 GB with PyTorch, 1.5 GB CPU-only). Safe to run again.
set -euo pipefail
VENV="${VENV:-$HOME/ooa-train-venv}"
CPU=0
[ "${1:-}" = "--cpu" ] && CPU=1

echo "== 1/4 GPU check"
if [ "$CPU" = "1" ]; then
  echo "CPU-only setup (no NVIDIA GPU needed): $(nproc) CPU cores"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi does not work here. In WSL: install/update the NVIDIA driver in WINDOWS (not inside Ubuntu),"
  echo "restart Windows, open Ubuntu again and rerun this script."
  echo "No NVIDIA GPU at all? Run the CPU-only setup instead (add  -s -- --cpu  after bash, see the top of this file)."
  exit 1
fi

echo "== 2/4 system packages (asks for your Ubuntu password; may wait for Ubuntu's own first updates)"
sudo apt-get -o DPkg::Lock::Timeout=900 update -y
sudo apt-get -o DPkg::Lock::Timeout=900 install -y python3 python3-venv python3-dev build-essential git

echo "== 3/4 Python environment in $VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# the PyTorch version nanochat (commit 92d63d4) was written for; the Linux wheel includes CUDA
if [ "$CPU" = "1" ]; then
  "$VENV/bin/pip" install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu
else
  "$VENV/bin/pip" install "torch==2.9.1"
fi
"$VENV/bin/pip" install rustbpe tiktoken kernels filelock pyarrow psutil requests wandb numpy jsonschema

echo "== 4/4 compiler test (about a minute)"
OOA_CPU="$CPU" "$VENV/bin/python" - <<'EOF'
import os, torch
if os.environ.get("OOA_CPU") == "1":
    f = torch.compile(lambda x: (x * 2).sin().sum())
    f(torch.randn(1024))
    print("PyTorch", torch.__version__, "| CPU compiler test: OK")
    raise SystemExit(0)
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
