#!/bin/bash
# Bring a prover endpoint up or down.
#
# BFS needs llama.cpp specifically: its search ranks candidates by cumulative
# token log-probability, and LM Studio does not return logprobs on the
# completion endpoint (verified 2026-07-29). Goedel does not need them — the
# whole-proof loop reads only `content`, `finish_reason`, and
# `usage.completion_tokens`, all of which LM Studio provides — so it can be
# served from anywhere, including another machine on your network. Point the
# runner at it with `--whole-proof-url <base>/v1`.
#
# GGUF files are looked up under $HOME/.cache/lm-studio/models (LM Studio's
# download layout); set MODELS below or symlink your own downloads there.
#
#   ./scripts/provers.sh up bfs        local llama-server :8080
#   ./scripts/provers.sh up goedel     local llama-server :8081
#   ./scripts/provers.sh down bfs
#   ./scripts/provers.sh status
set -uo pipefail

MODELS="$HOME/.cache/lm-studio/models"
BFS_GGUF="$MODELS/DevQuasar/ByteDance-Seed.BFS-Prover-V2-7B-GGUF/ByteDance-Seed.BFS-Prover-V2-7B.Q8_0.gguf"
GOEDEL_GGUF="$MODELS/NikolayKozloff/Goedel-Prover-V2-8B-Q8_0-GGUF/goedel-prover-v2-8b-q8_0.gguf"
REMOTE="${LM_STUDIO_BASE_URL:-}"   # optional LM Studio host for Goedel, e.g. http://host:1234/v1

up() {
  case "$1" in
    # 4096 is this checkpoint's trained context; asking for more makes
    # llama.cpp warn and clamp, so it is set explicitly rather than inherited.
    bfs)    model="$BFS_GGUF";    port=8080; ctx=4096  ;;
    goedel) model="$GOEDEL_GGUF"; port=8081; ctx=16384 ;;
    *) echo "unknown prover: $1" >&2; return 2 ;;
  esac
  if curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "$1 already up on :$port"; return 0
  fi
  nohup llama-server -m "$model" --ctx-size "$ctx" --parallel 1 -ngl 99 \
    --host 127.0.0.1 --port "$port" --metrics > "/tmp/llama_$1.log" 2>&1 &
  for _ in $(seq 1 60); do
    sleep 2
    if curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "$1 up on :$port (ctx $ctx)"; return 0
    fi
  done
  echo "$1 did not come up — see /tmp/llama_$1.log" >&2; return 1
}

down() {
  case "$1" in
    bfs)    pattern="BFS-Prover"    ;;
    goedel) pattern="goedel-prover" ;;
    *) echo "unknown prover: $1" >&2; return 2 ;;
  esac
  if pkill -f "llama-server.*$pattern" 2>/dev/null; then
    echo "$1 down"
  else
    echo "$1 was not running"
  fi
}

status() {
  for entry in 8080:bfs 8081:goedel; do
    port="${entry%%:*}"; name="${entry##*:}"
    if curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "  :$port  $name    up"
    else
      echo "  :$port  $name    down"
    fi
  done
  [ -n "$REMOTE" ] || return 0
  echo "  remote (LM Studio — serves Goedel; cannot serve BFS, no logprobs):"
  if curl -s -m 4 "$REMOTE/models" >/dev/null 2>&1; then
    echo "    $REMOTE  reachable"
  else
    echo "    $REMOTE  unreachable"
  fi
}

case "${1:-status}" in
  up)     up "${2:-}" ;;
  down)   down "${2:-}" ;;
  status) status ;;
  *) echo "usage: $0 {up|down} {bfs|goedel} | status" >&2; exit 2 ;;
esac
