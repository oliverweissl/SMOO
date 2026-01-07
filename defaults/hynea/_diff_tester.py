from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from itertools import product
from time import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from src import SMOO, TEarlyTermCallable
from src.manipulator.diffusion_manipulator import (
    DiffusionCandidate,
    DiffusionCandidateList,
    SiTManipulator,
)
from src.objectives import CriterionCollection
from src.optimizer import Optimizer, PymooOptimizer
from src.sut import ClassifierSUT

from ._experiment_config import ExperimentConfig


class DiffTester(SMOO):
    """A tester class for mimicry with diffusion models."""

    _manipulator: SiTManipulator
    _optimizer: PymooOptimizer
    _sut: ClassifierSUT
    _config: ExperimentConfig

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: SiTManipulator,
        optimizer: Optimizer,
        objectives: CriterionCollection,
        config: ExperimentConfig,
        early_termination: TEarlyTermCallable,
    ):
        """
        Initialize the Diffusion-based Tester (mimicry with diffusion)..

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
            restrict_classes=config.restrict_classes,
            use_wandb=False,
            early_termination=early_termination,
        )
        self._config = config

    def test(self) -> None:
        """Start the diffusion-based testing."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for class_id, sample_idx in product(
            self._config.classes, range(self._config.samples_per_class)
        ):
            logging.info(f"Test class {class_id}, sample idx {sample_idx}.")

            """Get initial origin and target candidates."""
            origin, y_hat, origin_image = self._find_valid_candidate(class_id, is_origin=True)
            origin_batch = origin_image.expand(
                self._optimizer.get_x_current().shape[0], *origin_image.shape[1:]
            )  # These are just memory views: if inplace function needed do .clone()
            _, second, *_ = torch.argsort(y_hat[0], descending=True)

            target_class = int(second.item()) if self._config.run_targeted else class_id
            target, _, target_image = self._find_valid_candidate(target_class, is_origin=False)
            candidates = DiffusionCandidateList(origin, target)
            global_start = time()  # Stores the start time of the current experiment.

            """We run the optimization for one output."""
            res = self._optimization_loop(candidates, origin_batch, [class_id, target_class])

            """Save data."""
            log_dir = os.path.join(
                script_dir, f"runs/class_{class_id}_{self._config.save_as}_{time()}"
            )
            os.makedirs(log_dir, exist_ok=True)

            stats = {
                "runtime": time() - global_start,
                "y_hat": y_hat.cpu().squeeze().tolist(),
                "budget_used": res.budget_used,
            }
            df = pd.DataFrame(res.generation_history)
            df.to_csv(log_dir + "/data.csv", index=False)

            if res.termination_selection is not None:
                """Here we save all elements that satisfy a termination condition."""
                indices = np.arange(res.termination_selection.shape[0])[res.termination_selection]
                for ind in indices:
                    if res.xs is not None:
                        self._save_tensor_as_image(res.xs[ind], log_dir + f"/best_{ind}.png")
                    if res.predictions is not None:
                        stats[f"best_{ind}_y_hat"] = res.predictions[ind].tolist()
                    stats[f"best_{ind}_solution"] = res.solutions[ind].tolist()
                    stats[f"best_{ind}_fitness"] = [
                        res.generation_history[-1][n][ind] for n in self._objectives.names
                    ]
            else:
                """If no termination condition was met, we save the best candidates."""
                for i, bc in enumerate(self._optimizer.best_candidates):
                    self._save_tensor_as_image(bc.data[0], log_dir + f"/best_{i}.png")
                    stats[f"best_{i}_y_hat"] = bc.data[1].tolist()
                    stats[f"best_{i}_solution"] = bc.solution.tolist()
                    stats[f"best_{i}_fitness"] = list(bc.fitness)
            """Here we save the standard images for easy comparison."""
            self._save_tensor_as_image(origin_image, log_dir + f"/origin_{class_id}.png")
            self._save_tensor_as_image(target_image, log_dir + f"/taget_{target_class}.png")

            with open(f"{log_dir}/stats.json", "w") as f:
                f.write(json.dumps(stats))

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}"
            )

    def _optimization_loop(
        self,
        candidates: DiffusionCandidateList,
        origin_batch: Tensor,
        class_pair: list[int],
    ) -> _OptimResults:
        """
        The optimization loop used.

        :param candidates: The candidates to optimize with.
        :param origin_batch: The origin image batch for comparison.
        :param class_pair: The class information of both origin and target.

        :returns: The optimization Results.
        """

        start_idx = 0  # The start index for solution chunks.
        budget_used: int = 0  # Computational budget measured by SUT evals.
        solution_cache = np.zeros(self._config.solution_shape)  # An empty solution array.

        all_gen_data: list[dict[str, Any]] = (
            []
        )  # Stores fitness values of individuals per generation.
        term_selection: Optional[NDArray] = None  # Which outputs triggered the early termination.
        terminate_early = False
        xs = torch.empty(0)  # Empty variable to allow static checkers to see it is there.
        predictions = torch.empty(0)  # Empty variable to allow static checkers to see it is there.

        """Start population based optimization."""
        for f, solution_size in enumerate(self._config.optimizer_schedule):
            """Adapt problem to fit solution chunk."""
            sol_chunk = solution_cache[:, :, start_idx : start_idx + solution_size]
            # We sample for the size of the solution chunk with some custom distribution.
            self._optimizer.update_problem(
                solution_shape=(2, solution_size),
                sampling=np.random.beta(a=1, b=5, size=sol_chunk.shape),
            )
            logging.info("=" * 50)
            logging.info(
                f"Optimizing solution chunk {f + 1}/{len(self._config.optimizer_schedule)}"
            )
            intermediate_generations = self._config.generations // len(
                self._config.optimizer_schedule
            )
            for i in range(1, intermediate_generations + 1):
                gen_start = time()
                logging.info("_" * 50)
                logging.info(f"Generation {i} start.")

                solution_cache[:, :, start_idx : start_idx + solution_size] = (
                    self._optimizer.get_x_current()
                )

                """Here we reverse the tensor as the first diffusion steps would be on the back of the weights."""
                sol_tensor = torch.as_tensor(solution_cache)
                sols = sol_tensor if self._config.reverse_schedule else sol_tensor.flip(-1)
                xw = sols[:, 0, ...] * self._config.use_diffusion_manipulation.real
                yw = sols[:, 1, ...] * self._config.use_condition_manipulation.real
                xs_new = self._manipulator.manipulate(candidates, xw, yw)

                """We predict the label from the mixed images."""
                xs = self._manipulator.get_images(xs_new)
                predictions = self._process(xs)
                budget_used += xs.shape[0]  # add budget based on how many images are evaluated.

                self._objectives.evaluate_all(
                    images=[origin_batch, xs],
                    logits=predictions,
                    label_targets=class_pair,
                    solution_archive=list(),
                    batch_dim=0,
                )
                results = self._objectives.results

                row = {
                    "generation": i,
                }
                row |= results

                all_gen_data.append(row)
                xsc = np.ascontiguousarray(xs.cpu())
                self._optimizer.assign_fitness(
                    [np.asarray(f) for f in results.values()], xsc, predictions.numpy()
                )
                self._optimizer.update()

                terminate_early, term_selection = self._early_termination(results)
                logging.info(f"Generation {i} done in {time() - gen_start}.")
                if terminate_early:
                    logging.info(
                        f"Early termination triggered at generation {i} by {np.sum(term_selection)} individuals."
                    )
                    break
                else:
                    del xw, yw, xsc, xs_new, sol_tensor
                    self._cleanup()  # Free up some memory

            """Assign best candidates uniformly to the cached solution."""
            stack = np.stack(self._optimizer.best_solutions_reshaped, axis=0)
            batch_size = self._optimizer.get_x_current().shape[0]
            num_cand = len(self._optimizer.best_candidates)
            # N x 2 x 50 -> B x 2 x 50
            solutions = np.tile(stack, ((batch_size + num_cand - 1) // num_cand, 1, 1))[:batch_size]

            solution_cache[:, :, start_idx : start_idx + solution_size] = solutions
            start_idx += solution_size
            if terminate_early:
                break

        results = _OptimResults(
            generation_history=all_gen_data,
            budget_used=budget_used,
            termination_selection=term_selection,
            solutions=solution_cache,
            xs=xs,
            predictions=predictions,
        )
        return results

    def _find_valid_candidate(
        self, class_id: int, is_origin: bool = False
    ) -> tuple[DiffusionCandidate, Tensor, Tensor]:
        """
        Sample single candidates that are valid to the SUT.

        :param class_id: The class ID.
        :param is_origin: Whether the candidate is a origin candidate.
        :returns: The DiffusionCandidate and the prediction of the SUT and the generated Image.
        """
        while True:
            xt, emb = self._manipulator.get_diff_steps([class_id])
            image = self._manipulator.get_images(xt[-1])
            y_hat = self._process(image)
            if torch.argmax(y_hat) == class_id:
                break
            logging.warning(
                f"Failed to find candidate for {class_id}, predicted {torch.argmax(y_hat)}"
            )
        candidate = DiffusionCandidate(xt.squeeze(), emb, is_origin=is_origin)
        return candidate, y_hat, image


@dataclass
class _OptimResults:
    """A storage class for less convoluted return statements."""

    generation_history: list[dict[str, Any]]  # The history of optimization results.
    budget_used: int  # The amount of SUT evaluations used.
    termination_selection: Optional[NDArray]  # The solutions that triggered early termination.
    solutions: NDArray  # The solution interpolation weights.
    xs: Optional[Tensor]  # The last generated images.
    predictions: Optional[Tensor]  # The last predictions.
