import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src import SMOO, TEarlyTermCallable
from src.manipulator.pertubation_manipulator import (
    MultimodalManipulator,
    PerturbCandidateList,
)
from src.objectives import CriterionCollection
from src.optimizer import Optimizer
from src.sut import VLMSUT

from ._experiment_config import ExperimentConfig
from ._helpers import (
    active_solution_shape,
    build_population_candidates,
    evaluate_baseline,
    extract_json_array,
    extract_target_objects,
    load_sample,
    prepare_bbox_pairs,
    save_baseline_fail,
    save_best_result,
    split_manipulation_results,
)
from ._prompts import DETECTION_PROMPT
from ._qwen3_embedding import Qwen3EmbeddingInstance

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
        """Initialize the Multi-Modal Tester."""
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
        self._text_embedder = Qwen3EmbeddingInstance(seed=config.seed)

    def test(self) -> None:
        """Start the multi-modal testing."""
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        random.seed(self._config.seed)
        np.random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)

        output_dir = (
            script_dir / "results" / self._config.save_as / OUTPUT_BASE_DIRS[self._config.mode]
        )
        selection_dir = script_dir / self._config.results_dir / self._config.selection_dir
        all_samples = self.get_all_sample_folders(selection_dir)
        if not all_samples:
            logging.info("Nothing to process. Exiting.")
            return
        logging.info("Found %d valid sample folders.", len(all_samples))

        ############# Check what samples need to be optimized still
        pending = []
        skipped = 0
        for folder_path, category, folder_id in all_samples:
            if self.is_already_processed(category, folder_id, str(output_dir)):
                skipped += 1
            else:
                pending.append((folder_path, category, folder_id))

        if not pending:
            logging.info("All samples already processed. Exiting.")
            return
        logging.info("Skipping %d already-processed samples. %d remaining.", skipped, len(pending))

        image_dim = 0
        text_dim = 0
        for inner in getattr(self._manipulator, "_manipulators", []):
            if hasattr(inner, "perturbations"):
                image_dim = len(inner.perturbations)
            if hasattr(inner, "obj_pertubations") and hasattr(inner, "prompt_pertubations"):
                text_dim = len(inner.obj_pertubations) + len(inner.prompt_pertubations)
        solution_shape = self._config.solution_shape or active_solution_shape(
            self._config.mode, image_dim, text_dim
        )
        if self._optimizer.n_var != int(np.prod(solution_shape)):
            update_problem = getattr(self._optimizer, "update_problem", None)
            if callable(update_problem):
                update_problem(solution_shape)
            else:
                raise ValueError(
                    f"Optimizer genome size {self._optimizer.n_var} does not match MMM solution shape {solution_shape}."
                )

        ############# Run optimization for all samples.
        for sample_idx, (folder_path, category, folder_id) in enumerate(pending, start=1):
            sample_label = f"{category}/{folder_id}"
            logging.info("SAMPLE %d/%d %s", sample_idx, len(pending), sample_label)

            sample_data = load_sample(folder_path, max_resolution=self._config.max_resolution)

            sample_data["category"] = category
            sample_data["folder_id"] = folder_id

            baseline_iou = evaluate_baseline(self._sut, sample_data)
            sample_data["baseline_iou"] = baseline_iou
            sample_output_dir = os.path.join(str(output_dir), category, folder_id)

            if baseline_iou < self._config.baseline_iou:
                logging.info(
                    "%s baseline IoU=%.5f < %.2f",
                    sample_label,
                    baseline_iou,
                    self._config.baseline_iou,
                )
                save_baseline_fail(sample_output_dir, baseline_iou, sample_data)
                continue

            logging.info("%s baseline IoU=%.5f", sample_label, baseline_iou)
            self._optimizer.reset()

            early_stop_generation: int | None = None
            sample_start = time.time()

            for generation in range(self._config.generations):
                genomes = self._optimizer.get_x_current()
                candidates = build_population_candidates(
                    genomes,
                    sample_data,
                    prompt=DETECTION_PROMPT,
                    mode=self._config.mode,
                    image_dim=image_dim,
                    text_dim=text_dim,
                )
                fitness, objective_results, artifacts = self.evaluate_population(
                    candidates,
                    sample_data,
                )
                self._optimizer.assign_fitness(fitness, artifacts)

                terminate_early, _ = self._early_termination(objective_results)
                if terminate_early:
                    early_stop_generation = generation + 1
                    break

                if generation + 1 < self._config.generations:
                    self._optimizer.update()

            if not self._optimizer.best_candidates:
                raise RuntimeError(f"Optimizer produced no best candidates for {sample_label}.")

            best_candidate = min(
                self._optimizer.best_candidates,
                key=lambda candidate: np.linalg.norm(
                    np.asarray(candidate.fitness, dtype=np.float64)
                ),
            )
            save_best_result(
                sample_output_dir,
                sample_data,
                best_candidate,
                runtime=time.time() - sample_start,
                generations_completed=early_stop_generation or self._config.generations,
                early_stop_generation=early_stop_generation,
            )
            self._cleanup()

    def evaluate_population(
        self,
        candidates: PerturbCandidateList,
        sample_data: dict[str, Any],
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray], list[dict[str, Any]]]:
        """Evaluate one optimizer population and return objective arrays plus per-candidate artifacts."""
        manipulated = self._manipulator.manipulate(candidates)
        if not isinstance(manipulated, tuple):
            raise TypeError(
                f"MultimodalManipulator.manipulate must return a tuple of modality outputs, got {type(manipulated).__name__}."
            )
        images_np, prompts = split_manipulation_results(manipulated)
        images_pil = [Image.fromarray(image.astype(np.uint8)) for image in images_np]

        start_time = time.time()
        responses = self._sut.process_input((images_pil, prompts))
        elapsed = time.time() - start_time
        if len(responses) != len(images_pil):
            raise ValueError(
                f"VLM returned {len(responses)} responses for {len(images_pil)} input candidates."
            )

        clean_tensor = self._image_to_tensor(sample_data["clean_image_pil"])
        objective_values = {name: [] for name in self._objectives.names}
        artifacts: list[dict[str, Any]] = []

        for image_np, image_pil, prompt, response in zip(images_np, images_pil, prompts, responses):
            parsed_preds = extract_json_array(response)
            prompt_objects = extract_target_objects(prompt)
            pred_boxes, gt_boxes = prepare_bbox_pairs(
                sample_data["gt_bboxes"],
                parsed_preds,
                sample_data["orig_dims"][0],
                sample_data["orig_dims"][1],
                prompt_objects,
                self._sut.coord_scale,
                self._sut.bbox_order,
            )
            adv_tensor = self._image_to_tensor(image_np)
            self._objectives.evaluate_all(
                boxes=[pred_boxes, gt_boxes],
                images=[clean_tensor.unsqueeze(0), adv_tensor.unsqueeze(0)],
                embeddings=[
                    self._text_embedder.run_inference(sample_data["objects"]),
                    self._text_embedder.run_inference(prompt_objects),
                ],
            )

            per_candidate_results = {
                name: value for name, value in self._objectives.results.items()
            }
            missing_names = [
                name for name in self._objectives.names if name not in per_candidate_results
            ]
            if missing_names:
                raise KeyError(f"Missing objective results for: {missing_names}")

            for name, value in per_candidate_results.items():
                objective_values[name].append(value)

            artifacts.append(
                {
                    "image": image_pil,
                    "prompt": prompt,
                    "prompt_objects": prompt_objects,
                    "response": response,
                    "parsed_predictions": parsed_preds,
                    "results": per_candidate_results,
                    "runtime_seconds": elapsed / len(images_np),
                }
            )

        return (
            tuple(
                np.asarray(objective_values[name], dtype=np.float64)
                for name in self._objectives.names
            ),
            {
                name: np.asarray(values, dtype=np.float64)
                for name, values in objective_values.items()
            },
            artifacts,
        )

    @staticmethod
    def _image_to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
        image_array = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else image
        return torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

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
