#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python -c 'from initialize.data_selector import DataSelector; DataSelector(dataset_kind="imagenet_det").run_selection()'
python -c 'from initialize.data_selector import DataSelector; DataSelector(dataset_kind="udacity").run_selection()'
