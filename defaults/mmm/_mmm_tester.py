import logging
import os
import random
import time
from pathlib import Path

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

    _sut: VLMSUT
    _manipulator: MultimodalManipulator

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
        """Initialize the Multi-Modal Tester.

        :param sut: VLM system under test.
        :param manipulator: Multimodal perturbation manipulator.
        :param optimizer: Population-based optimizer.
        :param objectives: Objective collection used for scoring.
        :param config: MMM experiment configuration.
        :param early_termination: Early stopping callback.
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
        self._text_embedder = Qwen3EmbeddingInstance(seed=config.seed)

    def test(self) -> None:
        """Start the multi-modal testing.

        :raises ValueError: If the optimizer configuration does not match the MMM search space.
        """
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

        image_dim = self._manipulator.image_dim()
        text_dim = self._manipulator.text_dim()
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

        for sample_idx, (folder_path, category, folder_id) in enumerate(pending, start=1):
            sample_label = f"{category}/{folder_id}"
            logging.info("SAMPLE %d/%d %s", sample_idx, len(pending), sample_label)

            sample = load_sample(
                folder_path,
                max_resolution=self._config.max_resolution,
                category=category,
                folder_id=folder_id,
            )

            baseline_iou = evaluate_baseline(self._sut, sample)
            sample_output_dir = os.path.join(str(output_dir), category, folder_id)

            if baseline_iou < self._config.baseline_iou:
                logging.info(
                    "%s baseline IoU=%.5f < %.2f",
                    sample_label,
                    baseline_iou,
                    self._config.baseline_iou,
                )
                save_baseline_fail(sample_output_dir, sample)
                continue

            logging.info("%s baseline IoU=%.2f", sample_label, baseline_iou)
            self._optimizer.reset()

            early_stop_generation: int | None = None
            sample_start = time.time()

            for generation in range(self._config.generations):
                genomes = self._optimizer.get_x_current()
                candidates = build_population_candidates(
                    genomes,
                    sample,
                    prompt=DETECTION_PROMPT,
                    mode=self._config.mode,
                    image_dim=image_dim,
                    text_dim=text_dim,
                )
                candidates = self.evaluate_population(candidates)
                objective_results = candidates.get_objective_values(self._objectives.names)
                self._optimizer.assign_fitness(objective_results, candidates)

                for cand in self._optimizer.best_candidates:
                    terminate_early, _ = self._early_termination(cand.data.objective_values)
                    if terminate_early:
                        early_stop_generation = generation + 1
                        break
                if early_stop_generation is not None:
                    break

                if generation + 1 < self._config.generations:
                    self._optimizer.update()

            best_candidate = min(
                self._optimizer.best_candidates,
                key=lambda candidate: float(
                    np.linalg.norm(np.asarray(candidate.fitness, dtype=np.float64))
                ),
            )
            save_best_result(
                sample_output_dir,
                sample,
                best_candidate,
                runtime=time.time() - sample_start,
                generations_completed=early_stop_generation or self._config.generations,
                early_stop_generation=early_stop_generation,
            )
            self._cleanup()

    def evaluate_population(self, candidates: PerturbCandidateList) -> PerturbCandidateList:
        """Evaluate one optimizer population and attach objective values to candidates.

        :param candidates: Current optimizer population.
        :returns: The manipulated candidates with objective values and VLM outputs attached.
        :raises ValueError: If the VLM response batch size does not match the candidate batch size.
        """
        manipulated = self._manipulator.manipulate(candidates)
        responses = self._sut.process_input((manipulated.image_arrays, manipulated.prompts))

        clean_tensor = self._image_to_tensor(manipulated[0].sample.clean_image_pil)

        for candidate, prompt, response in zip(manipulated, manipulated.prompts, responses):
            try:
                parsed_predictions = extract_json_array(response)
            except ValueError as e:
                logging.warning("Corrupted VLM JSON output. Response=%r", response[:400])
                candidate.fail_code = str(e)
                parsed_predictions = []

            try:
                prompt_objects = extract_target_objects(prompt)
            except ValueError as e:
                logging.warning(
                    "Corrupted VLM Label output; assigning penalty. Response=%r", response[:400]
                )
                candidate.fail_code = str(e)
                prompt_objects = []

            pred_boxes, gt_boxes = prepare_bbox_pairs(
                candidate.sample.ground_truth_boxes,
                candidate.sample.original_size,
                parsed_predictions,
                candidate.sample.target_objects,
                self._sut.coord_scale,
                self._sut.bbox_order,
            )
            original_text = ", ".join(candidate.sample.target_objects)
            perturbed_text = ", ".join(prompt_objects)
            original_embedding, _, _ = self._text_embedder.run_inference(original_text)
            perturbed_embedding, _, _ = self._text_embedder.run_inference(perturbed_text)
            adv_tensor = self._image_to_tensor(candidate.image_array)

            self._objectives.evaluate_all(
                boxes=[pred_boxes, gt_boxes],
                images=[clean_tensor.unsqueeze(0), adv_tensor.unsqueeze(0)],
                embeddings=[original_embedding.squeeze(), perturbed_embedding.squeeze()],
            )

            candidate.objective_values = self._objectives.results
            candidate.vlm_response = response
            candidate.parsed_predictions = parsed_predictions
            candidate.matched_pred_boxes = pred_boxes.tolist()
            candidate.prompt_objects = prompt_objects

        return manipulated

    @staticmethod
    def _image_to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
        image_array = (
            np.array(image.convert("RGB"))
            if isinstance(image, Image.Image)
            else np.asarray(image, dtype=np.uint8)
        )
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
        sample_folders.sort(key=lambda item: (item[1], int(item[2])))
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
