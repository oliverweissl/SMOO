#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python -c 'from initialize.data_selector import DataSelector; DataSelector(dataset_kind="bdd100k").run_selection()'
