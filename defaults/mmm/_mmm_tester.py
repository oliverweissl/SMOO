import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

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

            evaluate_baseline(self._sut, sample)
            sample_output_dir = os.path.join(str(output_dir), category, folder_id)

            if sample.baseline_fail_code is not None:
                logging.info("%s baseline failed with: ", sample.baseline_fail_code)
                save_baseline_fail(sample_output_dir, sample)
                continue

            if sample.baseline_iou < self._config.baseline_iou:
                logging.info(
                    "%s baseline IoU=%.5f < %.2f",
                    sample_label,
                    sample.baseline_iou,
                    self._config.baseline_iou,
                )
                save_baseline_fail(sample_output_dir, sample)
                continue

            logging.info("%s baseline IoU=%.2f", sample_label, sample.baseline_iou)
            self._optimizer.reset()

            clean_tensor = self._image_to_tensor(sample.clean_image_array)
            original_text = ", ".join(sample.target_objects)
            original_embedding, _, _ = self._text_embedder.run_inference(original_text)
            embedding_cache: dict[str, np.ndarray] = {original_text: original_embedding}

            early_stop_generation: int | None = None
            sample_start = time.time()

            for generation in range(self._config.generations):
                generation_start = time.perf_counter()

                genomes = self._optimizer.get_x_current()

                build_start = time.perf_counter()
                candidates = build_population_candidates(
                    genomes,
                    sample,
                    prompt=DETECTION_PROMPT,
                    mode=self._config.mode,
                    image_dim=image_dim,
                    text_dim=text_dim,
                )
                build_time = time.perf_counter() - build_start

                candidates, eval_timings = self.evaluate_population(
                    candidates,
                    clean_tensor=clean_tensor,
                    original_embedding=original_embedding,
                    embedding_cache=embedding_cache,
                )

                assign_start = time.perf_counter()
                objective_results = candidates.get_objective_values(self._objectives.names)
                self._optimizer.assign_fitness(objective_results, candidates)
                assign_time = time.perf_counter() - assign_start

                for cand in self._optimizer.best_candidates:
                    terminate_early, _ = self._early_termination(cand.data.objective_values)
                    if terminate_early:
                        early_stop_generation = generation + 1
                        break

                update_time = 0.0
                if early_stop_generation is None and generation + 1 < self._config.generations:
                    update_start = time.perf_counter()
                    self._optimizer.update()
                    update_time = time.perf_counter() - update_start

                total_time = time.perf_counter() - generation_start
                logging.info(
                    (
                        "%s gen=%d timings total=%.3fs build=%.3fs manipulate=%.3fs "
                        "vlm=%.3fs embed=%.3fs objectives=%.3fs assign=%.3fs update=%.3fs"
                    ),
                    sample_label,
                    generation + 1,
                    total_time,
                    build_time,
                    eval_timings["manipulate"],
                    eval_timings["vlm"],
                    eval_timings["embed"],
                    eval_timings["objectives"],
                    assign_time,
                    update_time,
                )

                if early_stop_generation is not None:
                    break

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

    def evaluate_population(
        self,
        candidates: PerturbCandidateList,
        *,
        clean_tensor: torch.Tensor,
        original_embedding: np.ndarray,
        embedding_cache: dict[str, np.ndarray],
    ) -> tuple[PerturbCandidateList, dict[str, float]]:
        """Evaluate one optimizer population and attach objective values to candidates.

        :param candidates: Current optimizer population.
        :param clean_tensor: Cached clean-image tensor for the current sample.
        :param original_embedding: Cached embedding of the original target-object string.
        :param embedding_cache: Per-sample cache for perturbed-text embeddings.
        :returns: The manipulated candidates and timing breakdown.
        :raises ValueError: If the VLM response batch size does not match the candidate batch size.
        """
        manipulate_start = time.perf_counter()
        manipulated = self._manipulator.manipulate(candidates)
        manipulate_time = time.perf_counter() - manipulate_start

        prompts = manipulated.prompts
        vlm_start = time.perf_counter()
        responses = self._sut.process_input((manipulated.image_arrays, prompts))
        vlm_time = time.perf_counter() - vlm_start

        embed_time = 0.0
        objective_time = 0.0
        clean_batch = clean_tensor.unsqueeze(0)
        original_embedding_vector = original_embedding.squeeze()

        for candidate, prompt, response in zip(manipulated, prompts, responses):
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
                self._sut.coord_scale,
                self._sut.bbox_order,
            )
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                matched_pred_boxes = np.zeros((0, 4), dtype=np.float64)
            else:
                ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
                for i, pred_box in enumerate(pred_boxes):
                    px1, py1, px2, py2 = pred_box
                    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
                    for j, gt_box in enumerate(gt_boxes):
                        gx1, gy1, gx2, gy2 = gt_box
                        inter_x1 = max(px1, gx1)
                        inter_y1 = max(py1, gy1)
                        inter_x2 = min(px2, gx2)
                        inter_y2 = min(py2, gy2)
                        inter_w = max(0.0, inter_x2 - inter_x1)
                        inter_h = max(0.0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h
                        gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                        denom = pred_area + gt_area - inter_area
                        ious[i, j] = inter_area / denom if denom > 0.0 else 0.0
                pred_indices, gt_indices = linear_sum_assignment(-ious)
                matched_pred_boxes = pred_boxes[pred_indices]
            perturbed_text = ", ".join(prompt_objects)

            embed_start = time.perf_counter()
            perturbed_embedding = embedding_cache.get(perturbed_text)
            if perturbed_embedding is None:
                perturbed_embedding, _, _ = self._text_embedder.run_inference(perturbed_text)
                embedding_cache[perturbed_text] = perturbed_embedding
            embed_time += time.perf_counter() - embed_start

            objective_start = time.perf_counter()
            adv_tensor = self._image_to_tensor(candidate.image_array)
            self._objectives.evaluate_all(
                boxes=[gt_boxes, pred_boxes],
                images=[clean_batch, adv_tensor.unsqueeze(0)],
                embeddings=[original_embedding_vector, perturbed_embedding.squeeze()],
            )
            objective_time += time.perf_counter() - objective_start

            candidate.objective_values = dict(self._objectives.results)
            candidate.vlm_response = response
            candidate.parsed_predictions = parsed_predictions
            candidate.matched_pred_boxes = matched_pred_boxes.tolist()
            candidate.prompt_objects = prompt_objects

        return manipulated, {
            "manipulate": manipulate_time,
            "vlm": vlm_time,
            "embed": embed_time,
            "objectives": objective_time,
        }

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
