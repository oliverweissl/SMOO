#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model> [gpu] [port] [extra runner args...]" >&2
  exit 1
fi

MODEL="$1"
shift
GPU="0"
PORT="8700"
if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  GPU="$1"
  shift
fi
if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  PORT="$1"
  shift
fi

mkdir -p "${REPO_ROOT}/logs"
source "${SCRIPT_DIR}/_vllm_server.sh"
vllm_model_id "$MODEL" >/dev/null

cleanup() {
  stop_vllm_server
}
trap cleanup EXIT

start_vllm_server "$MODEL" "$GPU" "$PORT" "$REPO_ROOT"

for MODE in multi image text driving; do
  python "${REPO_ROOT}/experiments/run.py" --vlm "$MODEL" --mode "$MODE" --served-port "$PORT" "$@"
done
