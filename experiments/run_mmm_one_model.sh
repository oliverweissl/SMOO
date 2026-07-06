#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model> [served_port] [extra runner args...]" >&2
  exit 1
fi

MODEL="$1"
shift
PORT="8700"
if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  PORT="$1"
  shift
fi

for MODE in multi image text; do
  python experiments/run.py --vlm "$MODEL" --mode "$MODE" --served-port "$PORT" "$@"
done
