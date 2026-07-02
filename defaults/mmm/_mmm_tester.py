from src import SMOO, TEarlyTermCallable
from src.manipulator.pertubation_manipulator import MultimodalManipulator
from src.objectives import CriterionCollection
from src.optimizer import Optimizer
from src.sut import VLMSUT

import os
from pathlib import Path
import torch
import random
import numpy as np
import logging
from ._experiment_config import ExperimentConfig

OUTPUT_BASE_DIRS = {
    "multi": "multimodal",
    "image": "unimodal/image",
    "text": "unimodal/text",
}

class MMMTester(SMOO):
    """Multi-Modal Manipulation Tester."""

    def __init__(
        self,
        *,
        sut: VLMSUT,
        manipulator: MultimodalManipulator,
        optimizer: Optimizer,
        objectives: CriterionCollection,
        config: ExperimentConfig,
        early_termination: TEarlyTermCallable,
    ):
        """
        Initialize the Multi-Modal Tester.

        :param sut: The system-under-test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives used for fitness calculation.
        :param config: The experiment config.
        :param early_termination: An optional early termination function.
        """
        super().__init__(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            restrict_classes=None,
            use_wandb=False,
            early_termination=early_termination,
        )
        self._config = config

    def test(self) -> None:
        """Start the multi-modal testing."""
        script_dir = os.path.dirname(os.path.abspath(__file__))

        random.seed(self._config.seed)
        np.random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)

        """Get samples to test with."""
        output_dir = os.path.join(script_dir, "results", self._config.save_as, OUTPUT_BASE_DIRS[self._config.mode])
        all_samples = self.get_all_sample_folders(Path(script_dir / self._config.results_dir / self._config.selection_dir))
        if not all_samples:
            logging.info("Nothing to process. Exiting.")
            return
        logging.info("Found %d valid sample folders.", len(all_samples))

        pending = []
        skipped = 0
        for folder_path, category, folder_id in all_samples:
            if self.is_already_processed(category, folder_id, output_dir):
                skipped += 1
            else:
                pending.append((folder_path, category, folder_id))

        if not pending:
            logging.info("All samples already processed. Exiting.")
            return
        logging.info("Skipping %d already-processed samples. %d remaining.", skipped, len(pending))

    @staticmethod
    def get_all_sample_folders(results_dir: str | Path) -> list[tuple[str, str, str]]:
        """Discover all valid sample folders under the three annotation categories.

        :param results_dir: Root results directory produced by the data selector.
        :returns: Sorted list of ``(folder_path, category, folder_id)`` tuples.
        """
        sample_folders = []
        for cat_rel in (os.path.join("single", "solo"), os.path.join("single", "multi"), "multi"):
            cat_abs = os.path.join(results_dir, cat_rel)
            if not os.path.isdir(cat_abs):
                continue
            for fn in os.listdir(cat_abs):
                fp = os.path.join(cat_abs, fn)
                if not os.path.isdir(fp) or not fn.isdigit():
                    continue
                if not os.path.exists(os.path.join(fp, "original.json")):
                    continue
                if not os.path.exists(os.path.join(fp, "data_point.JPEG")):
                    continue
                sample_folders.append((fp, cat_rel, fn))
        sample_folders.sort(key=lambda t: (t[1], int(t[2])))
        return sample_folders

    @staticmethod
    def is_already_processed(category: str, folder_id: str, output_base: str) -> bool:
        """Check whether a sample has already been processed by inspecting output files.

        :param category: Category relative path.
        :param folder_id: Numeric folder identifier string.
        :param output_base: Root output directory.
        :returns: ``True`` if a best-result or baseline-fail file exists for this sample.
        """
        out = os.path.join(output_base, category, folder_id)
        return os.path.exists(os.path.join(out, "best_result.json")) or os.path.exists(
            os.path.join(out, "baseline_fail.json")
        )