#!/usr/bin/env bash
set -euo pipefail

# Deterministic LM Studio reload for the Lean evaluation panel.
# This keeps the OpenAI-compatible endpoint at http://127.0.0.1:1234/v1
# but avoids GUI/JIT defaults by pinning identifiers and load params.

LMS="${LMS:-$HOME/.lmstudio/bin/lms}"

GOEDEL_MODEL="${GOEDEL_MODEL:-goedel-prover-v2-8b}"
GOEDEL_ID="${GOEDEL_ID:-goedel-prover-v2-8b}"
GOEDEL_CONTEXT="${GOEDEL_CONTEXT:-4096}"
GOEDEL_PARALLEL="${GOEDEL_PARALLEL:-1}"

BFS_MODEL="${BFS_MODEL:-bytedance-seed.bfs-prover-v2-7b}"
BFS_ID="${BFS_ID:-bytedance-seed.bfs-prover-v2-7b}"
BFS_CONTEXT="${BFS_CONTEXT:-2048}"
BFS_PARALLEL="${BFS_PARALLEL:-6}"

TTL="${TTL:-3600}"

if [[ ! -x "$LMS" ]]; then
  echo "lms CLI not found at $LMS" >&2
  exit 1
fi

"$LMS" server start >/dev/null || true

"$LMS" unload "$GOEDEL_ID" 2>/dev/null || true
"$LMS" unload "$BFS_ID" 2>/dev/null || true

"$LMS" load "$GOEDEL_MODEL" \
  --gpu max \
  --context-length "$GOEDEL_CONTEXT" \
  --parallel "$GOEDEL_PARALLEL" \
  --ttl "$TTL" \
  --identifier "$GOEDEL_ID" \
  -y

"$LMS" load "$BFS_MODEL" \
  --gpu max \
  --context-length "$BFS_CONTEXT" \
  --parallel "$BFS_PARALLEL" \
  --ttl "$TTL" \
  --identifier "$BFS_ID" \
  -y

"$LMS" ps
