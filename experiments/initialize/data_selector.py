import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io
import torch
from config.experiment import NUM_IMAGES, SEED
from config.prompts import DETECTION_PROMPT
from config.paths import (
    ANNOTATIONS_PATH,
    DATASET_PATH,
    MAT_FILE_PATH,
)
from config.paths import RESULTS_DIR as RESULTS_BASE_DIR
from PIL import Image
from tqdm import tqdm


def load_synset_to_label(mat_file_path):
    """Load the synset-to-human-label mapping from the ImageNet meta ``.mat`` file.

    :param mat_file_path: Path to the ImageNet ``meta.mat`` file.
    :returns: Dict mapping synset strings (e.g. ``"n01440764"``) to label strings.
    """
    meta = scipy.io.loadmat(mat_file_path)
    synsets = meta["synsets"]
    synset_to_label = {}
    for entry in synsets[0]:
        synset = entry[1][0]
        label = entry[2][0]
        synset_to_label[synset] = label
    return synset_to_label


def parse_annotation(xml_file, synset_to_label):
    """Parse an ImageNet PASCAL VOC XML annotation file.

    :param xml_file: Path to the XML annotation file.
    :param synset_to_label: Dict mapping synset IDs to human-readable labels.
    :returns: Tuple of (ground_truth dict, set of unique labels, instance count).
    """
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
        box = {
            "xmin": int(bndbox.find("xmin").text),
            "ymin": int(bndbox.find("ymin").text),
            "xmax": int(bndbox.find("xmax").text),
            "ymax": int(bndbox.find("ymax").text),
        }

        key = label
        if key in ground_truth:
            suffix = 1
            while f"{label}_{suffix}" in ground_truth:
                suffix += 1
            key = f"{label}_{suffix}"
        ground_truth[key] = box

    return ground_truth, unique_labels, instance_count


class DataSelector:
    """
    Selects a stratified subset of ImageNet DET val images based purely on
    annotation structure (no VLM involved).  All VLMs are later evaluated on
    the same subset; per-VLM baseline IoU is computed at search time.
    """

    def __init__(
        self,
        dataset_path: str = DATASET_PATH,
        annotations_path: str = ANNOTATIONS_PATH,
        mat_file_path: str = MAT_FILE_PATH,
        seed: int = SEED,
        results_dir: str = RESULTS_BASE_DIR,
    ):
        self.dataset_path = dataset_path
        self.annotations_path = annotations_path
        self.seed = seed
        self.results_dir = results_dir

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print("Loading Synset mappings...")
        self.synset_to_label = load_synset_to_label(mat_file_path)

    def scan_and_sort_candidates(self):
        """Scan the annotations directory and bucket images by category.

        :returns: Three shuffled lists (solo-instance, multi-instance, multi-class) of candidate dicts.
        """
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
            candidate = {"xml_file": xml_file, "image_file": image_file, "gt": gt_data}

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

        random.shuffle(single_class_multi_instance)
        random.shuffle(single_class_solo_instance)
        random.shuffle(multi_class_candidates)

        return single_class_solo_instance, single_class_multi_instance, multi_class_candidates

    def get_existing_progress(self, category):
        """Read existing saved selections to determine resume state.

        :param category: Category subdirectory name (e.g. ``"single/solo"``).
        :returns: Tuple of (next_index, set of already-saved image filenames).
        """
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
                    with open(result_file, "r") as f:
                        data = json.load(f)
                    if "image" in data:
                        completed_filenames.add(data["image"])
                    existing_indices.append(int(folder_name))
                except Exception:
                    pass

        next_index = max(existing_indices) + 1 if existing_indices else 1
        return next_index, completed_filenames

    def save_selection(self, cand, category, index):
        """Persist one selected sample: write ``original.json`` and copy the image.

        :param cand: Candidate dict with keys ``image_file`` and ``gt``.
        :param category: Category subdirectory name.
        :param index: Numeric folder index for this sample.
        """
        dir_path = os.path.join(self.results_dir, category, str(index))
        os.makedirs(dir_path, exist_ok=True)

        image_path = os.path.join(self.dataset_path, cand["image_file"])
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        object_names = set(k.split("_")[0] for k in cand["gt"].keys())
        objects_str = ", ".join(sorted(object_names))
        prompt = DETECTION_PROMPT.format(objects=objects_str)

        data = {
            "image": cand["image_file"],
            "prompt": prompt,
            "original_dims": [orig_w, orig_h],
            "seed": str(self.seed),
            "ground_truth": cand["gt"],
        }

        with open(os.path.join(dir_path, "original.json"), "w") as f:
            json.dump(data, f, indent=4)

        try:
            shutil.copy2(image_path, os.path.join(dir_path, "data_point.JPEG"))
        except Exception as e:
            print(f"Error copying {image_path}: {e}")

    def process_group(self, candidates, group_name, target_size):
        """Save selections from ``candidates`` until ``target_size`` samples exist for ``group_name``.

        :param candidates: Shuffled list of candidate dicts from :meth:`scan_and_sort_candidates`.
        :param group_name: Category directory name (e.g. ``"single/solo"``).
        :param target_size: Total number of samples required in this group.
        """
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
            image_path = os.path.join(self.dataset_path, cand["image_file"])
            if not os.path.exists(image_path):
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
        """Orchestrate the full selection pipeline across all three annotation categories."""
        solo_candidates, multi_inst_candidates, multi_class_candidates = (
            self.scan_and_sort_candidates()
        )

        self.process_group(solo_candidates, "single/solo", target_size=NUM_IMAGES)
        self.process_group(multi_inst_candidates, "single/multi", target_size=NUM_IMAGES)
        self.process_group(multi_class_candidates, "multi", target_size=NUM_IMAGES)

        print(f"\nSelection complete. Results saved in: {os.path.abspath(self.results_dir)}")


if __name__ == "__main__":
    selector = DataSelector(
        dataset_path=DATASET_PATH,
        annotations_path=ANNOTATIONS_PATH,
        mat_file_path=MAT_FILE_PATH,
        seed=SEED,
        results_dir=RESULTS_BASE_DIR,
    )
    selector.run_selection()
