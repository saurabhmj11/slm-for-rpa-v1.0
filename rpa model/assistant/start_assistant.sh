#!/usr/bin/env bash
# OOA Assistant launcher (Linux / macOS). Edit these paths, or set them in the environment before running.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

PYTHON="${PYTHON:-python3}"
MODEL_DIR="${MODEL_DIR:-$HERE/../nanochat_base}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$HERE/../nanochat}"
PLATFORM_DIR="${PLATFORM_DIR:-$HERE/../platform}"
CATALOG="${CATALOG:-$HERE/../sft/catalog.jsonl}"
[ -f "$CATALOG" ] || CATALOG="$HERE/../data/catalog/catalog.jsonl"
DATA_DIR="${DATA_DIR:-$HERE/assistant_data}"
PORT="${PORT:-8765}"
FAKE_MODEL="${FAKE_MODEL:-0}"     # 1 = try the app without the real model

[ -f "$PLATFORM_DIR/contracts/flow-schema/flow.v1.json" ] || { echo "Platform repo not found at $PLATFORM_DIR (set PLATFORM_DIR)"; exit 1; }
[ -f "$CATALOG" ] || { echo "Catalog not found at $CATALOG (set CATALOG)"; exit 1; }
EXTRA=()
if [ "$FAKE_MODEL" = "1" ]; then
  EXTRA=(--fake-model)
else
  [ -d "$MODEL_DIR/chatsft_checkpoints/d8" ] || { echo "Model not found at $MODEL_DIR/chatsft_checkpoints/d8 (set MODEL_DIR)"; exit 1; }
fi

exec "$PYTHON" "$HERE/app.py" --model-dir "$MODEL_DIR" --nanochat "$NANOCHAT_DIR" --platform "$PLATFORM_DIR" \
  --catalog "$CATALOG" --data-dir "$DATA_DIR" --port "$PORT" ${EXTRA[@]+"${EXTRA[@]}"}
