#!/bin/bash
# Optional step between 02 and 03.
# Requires Ollama running locally with the target model pulled.
# Produces methodology/auxiliary_files/homophone_mapping.json and synonym_mapping.json.
set -euo pipefail
cd "$(dirname "$0")"
echo "This can take some time if the generation chunks are small."
echo "Smaller chunks generally give better quality outputs."
echo "To change chunk size look at the config/experiment.py file."
PYTHONPATH=python -m initialize.build_label_mapping
PYTHONPATH=python -m initialize.generate_mappings "$@"
