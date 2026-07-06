DATASET_PATH = "dataset/2017/ILSVRC/Data/DET/val"
ANNOTATIONS_PATH = "dataset/2017/ILSVRC/Annotations/DET/val"
MAT_FILE_PATH = "dataset/2017/ILSVRC/devkit/data/meta_det.mat"

RESULTS_DIR = "results/selection"
SELECTION_CATEGORIES = ["single/solo", "single/multi", "multi"]

LABEL_MAPPING_FILE = "methodology/auxiliary_files/label_mapping.json"
HOMOPHONE_MAPPING_FILE = "methodology/auxiliary_files/homophone_mapping.json"
SYNONYM_MAPPING_FILE = "methodology/auxiliary_files/synonym_mapping.json"

OLLAMA_HOST = "http://localhost:11434"

OUTPUT_BASE_DIRS = {
    "multi": "multimodal",
    "image": "unimodal/image",
    "text": "unimodal/text",
}

PARETO_FILE = "pareto_front.json"
BEST_FILE = "best_result.json"
BEST_IMAGE_FILE = "best_result.png"
BASELINE_FAIL_FILE = "baseline_fail.json"
