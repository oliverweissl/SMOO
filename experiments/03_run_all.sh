#!/usr/bin/env bash
set -euo pipefail

MODELS=( qwen kimi intern gemma deepseek nemotron )

port_for_model() {
  case "$1" in
    qwen) echo 8700 ;;
    kimi) echo 8701 ;;
    intern) echo 8702 ;;
    gemma) echo 8703 ;;
    deepseek) echo 8704 ;;
    nemotron) echo 8705 ;;
    *)
      echo "Unknown model: $1" >&2
      exit 1
      ;;
  esac
}

GPU="${1:-0}"
if [ "$#" -gt 0 ]; then
  shift
fi

for MODEL in "${MODELS[@]}"; do
  PORT="$(port_for_model "$MODEL")"
  bash experiments/_run_one.sh "$MODEL" "$GPU" "$PORT" "$@"
done
