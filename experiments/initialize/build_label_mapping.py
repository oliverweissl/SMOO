"""Collect unique class labels from selection results and write label_mapping.json."""

import json
import os
import re

from config.paths import LABEL_MAPPING_FILE, RESULTS_DIR, SELECTION_CATEGORIES


def extract_base_label(key: str) -> str:
    """Strip a trailing numeric suffix (e.g. ``_1``) from a ground-truth key.

    :param key: Ground-truth key such as ``"dog"`` or ``"dog_2"``.
    :returns: Base label string without the numeric suffix.
    """
    return re.sub(r"_\d+$", "", key)


def collect_labels(results_dir: str = RESULTS_DIR) -> set[str]:
    """Walk the selection results directory and collect all unique base class labels.

    :param results_dir: Root directory containing category sub-folders with ``original.json`` files.
    :returns: Set of unique base label strings found across all selections.
    """
    labels: set[str] = set()
    for category in SELECTION_CATEGORIES:
        cat_dir = os.path.join(results_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for folder_name in sorted(os.listdir(cat_dir)):
            json_path = os.path.join(cat_dir, folder_name, "original.json")
            if not os.path.isfile(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)
            for key in data.get("ground_truth", {}):
                labels.add(extract_base_label(key))
    return labels


def main() -> None:
    """Collect unique labels from selections and write an empty ``label_mapping.json``."""
    labels = collect_labels()
    print(f"Found {len(labels)} unique class labels.")
    mapping = {label: [] for label in sorted(labels)}
    os.makedirs(os.path.dirname(LABEL_MAPPING_FILE), exist_ok=True)
    with open(LABEL_MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved label mapping to {LABEL_MAPPING_FILE}")


if __name__ == "__main__":
    main()
