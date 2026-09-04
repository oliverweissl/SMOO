#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPU="${1:-0}"
if [ "$#" -gt 0 ]; then shift; fi
PORT=8700
source "${SCRIPT_DIR}/_vllm_server.sh"
cleanup() { stop_vllm_server; }
trap cleanup EXIT
for MODEL in nemotron; do # gemma deepseek qwen kimi intern
  start_vllm_server "$MODEL" "$GPU" "$PORT" "$REPO_ROOT"
  python "${SCRIPT_DIR}/run_ablation.py" --model "$MODEL" --served-port "$PORT" "$@"
  stop_vllm_server
done
