"""Collect unique class labels from selection results and write label_mapping.json."""

import json
import re
from pathlib import Path

from config.paths import LABEL_MAPPING_FILE, RESULTS_DIR


def extract_base_label(key: str) -> str:
    """Strip a trailing numeric suffix (e.g. ``_1``) from a ground-truth key."""
    return re.sub(r"_\d+$", "", key)


def collect_labels(results_dir: str = RESULTS_DIR) -> set[str]:
    """Walk the selection directory recursively and collect all unique base class labels."""
    labels: set[str] = set()
    for json_path in sorted(Path(results_dir).rglob("original.json")):
        if not json_path.is_file():
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for key in data.get("ground_truth", {}):
            labels.add(extract_base_label(key))
    return labels


def main() -> None:
    """Collect unique labels from selections and write an empty ``label_mapping.json``."""
    labels = collect_labels()
    print(f"Found {len(labels)} unique class labels.")
    mapping = {label: [] for label in sorted(labels)}
    Path(LABEL_MAPPING_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved label mapping to {LABEL_MAPPING_FILE}")


if __name__ == "__main__":
    main()
