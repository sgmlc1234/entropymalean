#!/usr/bin/env bash
set -euo pipefail

# Direct Goedel serving without the LM Studio app server.
# Requires llama.cpp's standalone `llama-server`, e.g.:
#   brew install llama.cpp
#
# Evaluation env for this server:
#   LM_STUDIO_BASE_URL=http://127.0.0.1:1235/v1
#   EVAL_GOEDEL_V2_SLUG=goedel-prover-v2-8b
#   GOEDEL_PROMPT_STYLE=goedel_v2
#   GOEDEL_MAX_TOKENS=4096

LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
MODEL="${GOEDEL_GGUF:-$HOME/.cache/lm-studio/models/NikolayKozloff/Goedel-Prover-V2-8B-Q8_0-GGUF/goedel-prover-v2-8b-q8_0.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1235}"
ALIAS="${ALIAS:-goedel-prover-v2-8b}"
CTX_SIZE="${CTX_SIZE:-4096}"
PARALLEL="${PARALLEL:-1}"
GPU_LAYERS="${GPU_LAYERS:-999}"

if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
  echo "llama-server not found. Install it with: brew install llama.cpp" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "GGUF model not found: $MODEL" >&2
  exit 1
fi

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias "$ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --parallel "$PARALLEL" \
  --n-gpu-layers "$GPU_LAYERS" \
  --no-webui
