SEED = 42669
NUM_IMAGES = 100  # Total images per group (single, muli, mixed-multi)

# NSGA-II
POP_SIZE = 30
NUM_GENERATIONS = 15
BATCH_SIZE = 0  # 0 means all elements are taken, adjust if memory is limited.
BUDGET_MAX = 1.0
N_OBJ = 3  # Needs adjustment in _evaluator if changed!

# Baseline filter
BASELINE_IOU_MIN = 0.5

# Early stopping
EARLY_STOP_IOU_MAX = 0.35
EARLY_STOP_IMG_DIST_MAX = 0.1
EARLY_STOP_TXT_SIM_MIN = 0.70

# Image preprocessing
MAX_RESOLUTION = 1024

# Below 0.01 every perturbation is sub-perceptual (e.g. noise std < 0.031, blur radius < 2.1px).
# This saves quite a lot of compute!
MIN_PERTURBATION_SCALE = 0.01

# Mapping Generation
OLLAMA_MODEL = "gpt-oss:120b"
MAPPING_CHUNK_SIZE = 3
