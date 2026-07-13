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

case "$MODEL" in
  qwen) MODEL_ID="Qwen/Qwen3-VL-4B-Instruct" ;;
  kimi) MODEL_ID="moonshotai/Kimi-VL-A3B-Instruct" ;;
  intern) MODEL_ID="OpenGVLab/InternVL3_5-8B" ;;
  gemma) MODEL_ID="google/gemma-3-4b-it" ;;
  deepseek) MODEL_ID="deepseek-ai/deepseek-vl2-tiny" ;;
  nemotron) MODEL_ID="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" ;;
  *)
    echo "Unknown model: $MODEL" >&2
    exit 1
    ;;
esac

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1
CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL_ID" \
  --port "$PORT" \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --max-model-len 4096 \
  > "${REPO_ROOT}/logs/${MODEL}_server.log" 2>&1 &
SERVER_PID=$!

until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
  sleep 3
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: $MODEL server died" >&2; exit 1; }
done

for MODE in multi image text; do
  python "${REPO_ROOT}/experiments/run.py" --vlm "$MODEL" --mode "$MODE" --served-port "$PORT" "$@"
done
