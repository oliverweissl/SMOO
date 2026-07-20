from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.paths import UDACITY_DATASET_PATH, UDACITY_DOWNLOAD_ROOT, UDACITY_LABELS_PATH
else:
    from ..config.paths import UDACITY_DATASET_PATH, UDACITY_DOWNLOAD_ROOT, UDACITY_LABELS_PATH

DEFAULT_KAGGLE_DATASET = "alincijov/self-driving-cars"
EXPECTED_FILES = (
    Path(UDACITY_DATASET_PATH),
    Path(UDACITY_LABELS_PATH),
)


def verify_udacity_dataset() -> None:
    missing = [str(path) for path in EXPECTED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Udacity dataset preparation is incomplete. Missing: " + ", ".join(missing)
        )


def download_udacity_dataset(dataset_ref: str = DEFAULT_KAGGLE_DATASET, force: bool = False) -> None:
    root = Path(UDACITY_DOWNLOAD_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    if not force:
        try:
            verify_udacity_dataset()
            print(f"Udacity dataset already prepared at {root}")
            return
        except FileNotFoundError:
            pass

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_ref,
            "-p",
            str(root),
            "--unzip",
            "--force",
        ],
        check=True,
    )

    verify_udacity_dataset()
    print(f"Udacity dataset ready: images={UDACITY_DATASET_PATH}, labels={UDACITY_LABELS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the Udacity self-driving-car object-detection dataset."
    )
    parser.add_argument(
        "--dataset-ref",
        default=DEFAULT_KAGGLE_DATASET,
        help="Kaggle dataset reference to download (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the expected image root and labels CSV already exist.",
    )
    args = parser.parse_args()

    download_udacity_dataset(dataset_ref=args.dataset_ref, force=args.force)


if __name__ == "__main__":
    main()
