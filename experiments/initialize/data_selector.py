from __future__ import annotations

import csv
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy.io
import torch
from config.experiment import NUM_IMAGES, SEED
from config.paths import (
    ANNOTATIONS_PATH,
    BDD100K_DATASET_PATH,
    BDD100K_LABELS_PATH,
    DATASET_PATH,
    MAT_FILE_PATH,
    UDACITY_DATASET_PATH,
    UDACITY_LABELS_PATH,
)
from config.paths import RESULTS_DIR as RESULTS_BASE_DIR
from PIL import Image
from tqdm import tqdm

UDACITY_CLASS_ID_TO_LABEL = {
    "1": "car",
    "2": "truck",
    "3": "pedestrian",
    "4": "bicyclist",
    "5": "light",
}


def load_synset_to_label(mat_file_path):
    """Load the synset-to-human-label mapping from the ImageNet meta ``.mat`` file."""
    meta = scipy.io.loadmat(mat_file_path)
    synsets = meta["synsets"]
    synset_to_label = {}
    for entry in synsets[0]:
        synset = entry[1][0]
        label = entry[2][0]
        synset_to_label[synset] = label
    return synset_to_label


def _make_unique_gt_key(ground_truth: dict[str, dict[str, int]], label: str) -> str:
    key = label
    if key in ground_truth:
        suffix = 1
        while f"{label}_{suffix}" in ground_truth:
            suffix += 1
        key = f"{label}_{suffix}"
    return key


def _parse_int(value) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _normalize_box(box: dict[str, int]) -> dict[str, int] | None:
    xmin = _parse_int(box.get("xmin"))
    ymin = _parse_int(box.get("ymin"))
    xmax = _parse_int(box.get("xmax"))
    ymax = _parse_int(box.get("ymax"))
    if None in (xmin, ymin, xmax, ymax):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def parse_annotation(xml_file, synset_to_label):
    """Parse an ImageNet PASCAL VOC XML annotation file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ground_truth = {}
    unique_labels = set()

    objects = root.findall("object")
    instance_count = len(objects)

    for obj in objects:
        synset = obj.find("name").text
        label = synset_to_label.get(synset, synset)
        unique_labels.add(label)

        bndbox = obj.find("bndbox")
        box = _normalize_box(
            {
                "xmin": bndbox.find("xmin").text,
                "ymin": bndbox.find("ymin").text,
                "xmax": bndbox.find("xmax").text,
                "ymax": bndbox.find("ymax").text,
            }
        )
        if box is None:
            continue

        ground_truth[_make_unique_gt_key(ground_truth, label)] = box

    return ground_truth, unique_labels, instance_count


def normalize_bdd100k_stem(stem: str) -> str:
    """Normalize local BDD100K image stems to the JSON naming convention."""
    return stem[:-8] if stem.endswith("-0000100") else stem


def parse_bdd100k_labels(objects: list[dict]) -> tuple[dict[str, dict[str, int]], set[str], int]:
    """Normalize BDD100K ``labels`` entries into MMM ``ground_truth`` format."""
    ground_truth: dict[str, dict[str, int]] = {}
    unique_labels: set[str] = set()
    instance_count = 0

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("category", "")).strip()
        box2d = obj.get("box2d")
        if not label or not isinstance(box2d, dict):
            continue

        box = _normalize_box(
            {
                "xmin": box2d.get("x1"),
                "ymin": box2d.get("y1"),
                "xmax": box2d.get("x2"),
                "ymax": box2d.get("y2"),
            }
        )
        if box is None:
            continue

        ground_truth[_make_unique_gt_key(ground_truth, label)] = box
        unique_labels.add(label)
        instance_count += 1

    return ground_truth, unique_labels, instance_count


def parse_udacity_csv_rows(rows: list[dict]) -> tuple[dict[str, dict[str, int]], set[str], int]:
    """Normalize Udacity CSV rows into MMM ``ground_truth`` format."""
    ground_truth: dict[str, dict[str, int]] = {}
    unique_labels: set[str] = set()
    instance_count = 0

    for row in rows:
        if not isinstance(row, dict):
            continue

        label = str(
            row.get("label")
            or row.get("class")
            or row.get("class_name")
            or row.get("object")
            or UDACITY_CLASS_ID_TO_LABEL.get(str(row.get("class_id", "")).strip(), "")
        ).strip()
        if not label:
            continue

        box = _normalize_box(
            {
                "xmin": row.get("xmin") or row.get("x1"),
                "ymin": row.get("ymin") or row.get("y1"),
                "xmax": row.get("xmax") or row.get("x2"),
                "ymax": row.get("ymax") or row.get("y2"),
            }
        )
        if box is None:
            continue

        ground_truth[_make_unique_gt_key(ground_truth, label)] = box
        unique_labels.add(label)
        instance_count += 1

    return ground_truth, unique_labels, instance_count


class DataSelector:
    """Dataset-aware selection builder for MMM sample folders."""

    def __init__(
        self,
        dataset_kind: str = "bdd100k",
        dataset_path: str | None = None,
        annotations_path: str = ANNOTATIONS_PATH,
        mat_file_path: str = MAT_FILE_PATH,
        bdd100k_labels_path: str = BDD100K_LABELS_PATH,
        udacity_labels_path: str = UDACITY_LABELS_PATH,
        seed: int = SEED,
        results_dir: str = RESULTS_BASE_DIR,
    ):
        self.dataset_kind = dataset_kind
        if dataset_path is not None:
            self.dataset_path = dataset_path
        elif dataset_kind == "bdd100k":
            self.dataset_path = BDD100K_DATASET_PATH
        elif dataset_kind == "udacity":
            self.dataset_path = UDACITY_DATASET_PATH
        else:
            self.dataset_path = DATASET_PATH
        self.annotations_path = annotations_path
        self.bdd100k_labels_path = bdd100k_labels_path
        self.udacity_labels_path = udacity_labels_path
        self.seed = seed
        self.results_dir = results_dir
        self.synset_to_label = None

        self.random = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.dataset_kind == "imagenet_det":
            print("Loading Synset mappings...")
            self.synset_to_label = load_synset_to_label(mat_file_path)

    def scan_and_sort_candidates(self):
        """Scan the ImageNet DET annotations directory and bucket images by category."""
        if self.synset_to_label is None:
            raise ValueError("ImageNet selection requested without synset mappings.")

        print("Scanning dataset annotations...")
        xml_files = sorted([f for f in os.listdir(self.annotations_path) if f.endswith(".xml")])

        single_class_multi_instance = []
        single_class_solo_instance = []
        multi_class_candidates = []

        for xml_file in tqdm(xml_files, desc="Parsing XMLs"):
            xml_path = os.path.join(self.annotations_path, xml_file)
            gt_data, unique_labels, instance_count = parse_annotation(
                xml_path, self.synset_to_label
            )

            image_file = os.path.splitext(xml_file)[0] + ".JPEG"
            candidate = {
                "image_file": image_file,
                "image_path": os.path.join(self.dataset_path, image_file),
                "gt": gt_data,
            }

            if len(unique_labels) == 1:
                if instance_count >= 2:
                    single_class_multi_instance.append(candidate)
                else:
                    single_class_solo_instance.append(candidate)
            elif len(unique_labels) >= 2:
                multi_class_candidates.append(candidate)

        print(f"Found {len(single_class_multi_instance)} Single-Class (Multi-Instance) candidates.")
        print(f"Found {len(single_class_solo_instance)} Single-Class (Solo-Instance) candidates.")
        print(f"Found {len(multi_class_candidates)} Multi-Class candidates.")

        self.random.shuffle(single_class_multi_instance)
        self.random.shuffle(single_class_solo_instance)
        self.random.shuffle(multi_class_candidates)

        return single_class_solo_instance, single_class_multi_instance, multi_class_candidates

    def scan_bdd100k_candidates(self) -> list[dict]:
        """Collect all local BDD100K subset images with matched JSON labels."""
        with open(self.bdd100k_labels_path, "r", encoding="utf-8") as handle:
            labels = json.load(handle)

        label_map = {
            Path(item["name"]).stem: item
            for item in labels
            if isinstance(item, dict) and "name" in item
        }

        candidates: list[dict] = []
        image_paths = sorted(path for path in Path(self.dataset_path).iterdir() if path.is_file())
        for image_path in image_paths:
            image_stem = normalize_bdd100k_stem(image_path.stem)
            record = label_map.get(image_stem)
            if record is None:
                continue

            ground_truth, unique_labels, instance_count = parse_bdd100k_labels(record.get("labels", []))
            if not ground_truth or not unique_labels or instance_count <= 0:
                continue

            candidates.append(
                {
                    "image_file": image_path.name,
                    "image_path": str(image_path),
                    "source_name": record["name"],
                    "gt": ground_truth,
                }
            )

        print(f"Found {len(candidates)} BDD100K candidates with valid 2D boxes.")
        return candidates

    def scan_udacity_candidates(self) -> list[dict]:
        """Collect local Udacity images with valid CSV annotations."""
        image_root = Path(self.dataset_path)
        labels_path = Path(self.udacity_labels_path)

        missing = []
        if not image_root.exists():
            missing.append(f"image root: {image_root}")
        if not labels_path.exists():
            missing.append(f"labels CSV: {labels_path}")
        if missing:
            raise FileNotFoundError(
                "Udacity dataset is not prepared. Missing "
                + ", ".join(missing)
                + ". Run experiments/initialize/download_udacity.py first."
            )

        rows_by_image: dict[str, list[dict]] = {}
        with open(labels_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame_name = str(row.get("frame") or row.get("image") or row.get("filename") or "").strip()
                if not frame_name:
                    continue
                rows_by_image.setdefault(Path(frame_name).name, []).append(row)

        image_lookup = {path.name: path for path in sorted(image_root.rglob("*")) if path.is_file()}

        candidates: list[dict] = []
        for image_name, rows in rows_by_image.items():
            image_path = image_lookup.get(image_name)
            if image_path is None:
                continue

            ground_truth, unique_labels, instance_count = parse_udacity_csv_rows(rows)
            if not ground_truth or not unique_labels or instance_count <= 0:
                continue

            candidates.append(
                {
                    "image_file": image_path.name,
                    "image_path": str(image_path),
                    "source_name": image_name,
                    "gt": ground_truth,
                }
            )

        self.random.shuffle(candidates)
        print(f"Found {len(candidates)} Udacity candidates with valid 2D boxes.")
        return candidates

    def get_existing_progress(self, category):
        """Read existing saved selections to determine resume state."""
        category_dir = os.path.join(self.results_dir, category)
        if not os.path.exists(category_dir):
            return 1, set()

        completed_filenames = set()
        existing_indices = []

        for folder_name in os.listdir(category_dir):
            if not folder_name.isdigit():
                continue
            result_file = os.path.join(category_dir, folder_name, "original.json")
            if os.path.exists(result_file):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "image" in data:
                        completed_filenames.add(data["image"])
                    existing_indices.append(int(folder_name))
                except Exception:
                    pass

        next_index = max(existing_indices) + 1 if existing_indices else 1
        return next_index, completed_filenames

    def save_selection(self, cand, category, index):
        """Persist one selected sample: write ``original.json`` and copy the image."""
        dir_path = os.path.join(self.results_dir, category, str(index))
        os.makedirs(dir_path, exist_ok=True)

        image_path = cand["image_path"]
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        data = {
            "image": cand["image_file"],
            "original_dims": [orig_w, orig_h],
            "seed": str(self.seed),
            "ground_truth": cand["gt"],
        }
        if cand.get("source_name"):
            data["source_name"] = cand["source_name"]

        with open(os.path.join(dir_path, "original.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        try:
            shutil.copy2(image_path, os.path.join(dir_path, "data_point.JPEG"))
        except Exception as e:
            print(f"Error copying {image_path}: {e}")

    def process_group(self, candidates, group_name, target_size):
        """Save selections from ``candidates`` until ``target_size`` samples exist for ``group_name``."""
        print(f"\n--- Processing Group: {group_name} ---")

        next_save_index, completed_filenames = self.get_existing_progress(group_name)
        current_count = next_save_index - 1
        needed = target_size - current_count

        print(f"Status: {current_count}/{target_size} already completed. Need {needed} more.")

        if needed <= 0:
            print("Group already complete.")
            return

        saved = 0
        for cand in tqdm(candidates, desc=group_name):
            if saved >= needed:
                break
            if cand["image_file"] in completed_filenames:
                continue
            if not os.path.exists(cand["image_path"]):
                continue
            self.save_selection(cand, group_name, next_save_index)
            next_save_index += 1
            saved += 1

        if saved < needed:
            print(
                f"Warning: exhausted candidates for {group_name}. "
                f"Found {current_count + saved}/{target_size}."
            )

    def run_selection(self):
        """Orchestrate selection for the configured dataset."""
        if self.dataset_kind == "bdd100k":
            candidates = self.scan_bdd100k_candidates()
            self.process_group(candidates, "bdd100k", target_size=len(candidates))
        elif self.dataset_kind == "udacity":
            candidates = self.scan_udacity_candidates()
            self.process_group(candidates, "udacity", target_size=NUM_IMAGES)
        elif self.dataset_kind == "imagenet_det":
            solo_candidates, multi_inst_candidates, multi_class_candidates = (
                self.scan_and_sort_candidates()
            )
            self.process_group(solo_candidates, "single/solo", target_size=NUM_IMAGES)
            self.process_group(multi_inst_candidates, "single/multi", target_size=NUM_IMAGES)
            self.process_group(multi_class_candidates, "multi", target_size=NUM_IMAGES)
        else:
            raise ValueError(f"Unsupported dataset_kind: {self.dataset_kind}")

        print(f"\nSelection complete. Results saved in: {os.path.abspath(self.results_dir)}")


if __name__ == "__main__":
    selector = DataSelector(dataset_kind="bdd100k")
    selector.run_selection()
