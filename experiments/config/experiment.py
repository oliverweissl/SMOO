"""Data-selection and mapping-generation settings. Search hyperparameters are CLI flags of run.py."""

SEED = 42669
NUM_IMAGES = 100  # Total images per group (single, muli, mixed-multi)

# Baseline filter
BASELINE_IOU_MIN = 0.5

# Mapping Generation
OLLAMA_MODEL = "gpt-oss:120b"
MAPPING_CHUNK_SIZE = 3
