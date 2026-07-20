from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = str(ROOT_DIR / "dataset/2017/ILSVRC/Data/DET/val")
ANNOTATIONS_PATH = str(ROOT_DIR / "dataset/2017/ILSVRC/Annotations/DET/val")
MAT_FILE_PATH = str(ROOT_DIR / "dataset/2017/ILSVRC/devkit/data/meta_det.mat")

BDD100K_DATASET_PATH = str(ROOT_DIR / "dataset/bdd100k")
BDD100K_LABELS_PATH = str(ROOT_DIR / "dataset/det_v2_val_release.json")
UDACITY_DOWNLOAD_ROOT = str(ROOT_DIR / "dataset/udacity")
UDACITY_DATASET_PATH = str(ROOT_DIR / "dataset/udacity/images")
UDACITY_LABELS_PATH = str(ROOT_DIR / "dataset/udacity/labels_train.csv")

RESULTS_DIR = str(ROOT_DIR / "defaults/mmm/results/selection")
SELECTION_CATEGORIES = ["single/solo", "single/multi", "multi"]

LABEL_MAPPING_FILE = str(ROOT_DIR / "methodology/auxiliary_files/label_mapping.json")
HOMOPHONE_MAPPING_FILE = str(ROOT_DIR / "methodology/auxiliary_files/homophone_mapping.json")
SYNONYM_MAPPING_FILE = str(ROOT_DIR / "methodology/auxiliary_files/synonym_mapping.json")

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
