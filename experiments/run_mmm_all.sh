#!/usr/bin/env bash
set -euo pipefail

MODELS=(qwen kimi intern gemma deepseek)
declare -A PORTS=(
  [qwen]=8700
  [kimi]=8701
  [intern]=8702
  [gemma]=8703
  [deepseek]=8704
)

for MODEL in "${MODELS[@]}"; do
  bash experiments/run_mmm_one_model.sh "$MODEL" "${PORTS[$MODEL]}" "$@"
done
