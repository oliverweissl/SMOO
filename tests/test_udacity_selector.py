from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from initialize.data_selector import DataSelector, parse_udacity_csv_rows  # noqa: E402


class UdacitySelectorTests(unittest.TestCase):
    def test_parse_udacity_csv_rows_deduplicates_and_skips_invalid_boxes(self) -> None:
        ground_truth, labels, instance_count = parse_udacity_csv_rows(
            [
                {"label": "car", "xmin": "10", "ymin": "20", "xmax": "30", "ymax": "50"},
                {"label": "car", "xmin": "15", "ymin": "25", "xmax": "45", "ymax": "60"},
                {"class_id": "3", "xmin": "50", "ymin": "10", "xmax": "70", "ymax": "40"},
                {"label": "truck", "xmin": "10", "ymin": "10", "xmax": "10", "ymax": "20"},
                {"label": "", "xmin": "0", "ymin": "0", "xmax": "1", "ymax": "1"},
            ]
        )

        self.assertEqual(instance_count, 3)
        self.assertEqual(labels, {"car", "pedestrian"})
        self.assertEqual(list(ground_truth), ["car", "car_1", "pedestrian"])
        self.assertEqual(ground_truth["car"]["xmin"], 10)
        self.assertEqual(ground_truth["car_1"]["xmax"], 45)
        self.assertEqual(ground_truth["pedestrian"]["ymax"], 40)

    def test_udacity_selection_is_seeded_and_writes_exactly_100_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "images"
            image_root.mkdir(parents=True)
            labels_path = root / "labels_train.csv"
            results_a = root / "results_a"
            results_b = root / "results_b"

            rows = []
            for index in range(105):
                image_name = f"frame_{index:03d}.jpg"
                Image.new("RGB", (100, 80), color=(index % 255, 20, 40)).save(image_root / image_name)
                rows.append(
                    {
                        "frame": image_name,
                        "label": "car",
                        "xmin": "10",
                        "ymin": "10",
                        "xmax": "70",
                        "ymax": "60",
                    }
                )

            rows.append(
                {
                    "frame": "invalid.jpg",
                    "label": "car",
                    "xmin": "10",
                    "ymin": "10",
                    "xmax": "10",
                    "ymax": "60",
                }
            )

            with open(labels_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["frame", "label", "xmin", "ymin", "xmax", "ymax"])
                writer.writeheader()
                writer.writerows(rows)

            selector_a = DataSelector(
                dataset_kind="udacity",
                dataset_path=str(image_root),
                udacity_labels_path=str(labels_path),
                results_dir=str(results_a),
                seed=42669,
            )
            selector_b = DataSelector(
                dataset_kind="udacity",
                dataset_path=str(image_root),
                udacity_labels_path=str(labels_path),
                results_dir=str(results_b),
                seed=42669,
            )

            order_a = [cand["image_file"] for cand in selector_a.scan_udacity_candidates()]
            order_b = [cand["image_file"] for cand in selector_b.scan_udacity_candidates()]
            self.assertEqual(order_a, order_b)
            self.assertEqual(len(order_a), 105)

            selector_a.run_selection()

            saved = sorted((results_a / "udacity").glob("*/original.json"), key=lambda path: int(path.parent.name))
            self.assertEqual(len(saved), 100)

            with open(saved[0], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(set(payload), {"image", "original_dims", "seed", "ground_truth", "source_name"})
            self.assertEqual(payload["original_dims"], [100, 80])
            self.assertEqual(payload["seed"], "42669")
            self.assertEqual(list(payload["ground_truth"]), ["car"])
            self.assertTrue((saved[0].parent / "data_point.JPEG").exists())

    def test_udacity_selection_fails_clearly_when_dataset_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selector = DataSelector(
                dataset_kind="udacity",
                dataset_path=str(root / "missing_images"),
                udacity_labels_path=str(root / "missing_labels.csv"),
                results_dir=str(root / "results"),
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                selector.scan_udacity_candidates()

            self.assertIn("Run experiments/initialize/download_udacity.py first.", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
