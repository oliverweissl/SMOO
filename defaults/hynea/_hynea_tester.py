import json
import logging
import os
from itertools import product
from time import time
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from src import SMOO
from src.manipulator.diffusion_manipulator import (
    DiffusionCandidate,
    DiffusionCandidateList,
    LDMHyNeAManipulator,
    SitHyNeAManipulator,
)
from src.objectives import CriterionCollection
from src.optimizer import TorchModelOptimizer
from src.sut import BinaryClassifierSUT, ClassifierSUT, YoloSUT

from ._experiment_config import ExperimentConfig


class HyNeATester(SMOO):
    """A tester class that implements the HyNeA method."""

    _manipulator: LDMHyNeAManipulator | SitHyNeAManipulator
    _optimizer: TorchModelOptimizer
    _sut: ClassifierSUT
    _config: ExperimentConfig

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: LDMHyNeAManipulator | SitHyNeAManipulator,
        optimizer: TorchModelOptimizer,
        objectives: CriterionCollection,
        config: ExperimentConfig,
    ):
        """
        Initialize the HyNeA Tester.

        :param sut: The system-under-test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives used for fitness calculation.
        :param config: The experiment config.
        """

        super().__init__(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            restrict_classes=config.restrict_classes,
            use_wandb=False,
        )
        self._sut.gradient_checkpointing(enable=True)
        self._manipulator.gradient_checkpointing(enable=True)

        self._config = config

    def test(self) -> None:
        """
        Start the HyNeA-based testing.

        :raises NotImplementedError: This method is not implemented for a specific SUT.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for class_id, sample_idx in product(
            self._config.classes, range(self._config.samples_per_class)
        ):
            logging.info(f"Test class {class_id}, sample idx {sample_idx}.")
            cand_i, y0, i0 = self._find_valid_candidate(class_id, is_origin=True)
            initial_pred = torch.argsort(
                y0[0], descending=True
            )  # Only used in multi-class classification
            control = torch.zeros(1, *self._manipulator.control_shape, device=cand_i.xt.device)
            assert (
                control.shape == y0.shape
            ), f"Error in Control shape. Got {control.shape} instead of {y0.shape}."

            # Adapt the functionality of testing based on the SUT.
            if isinstance(self._sut, ClassifierSUT):
                target = int(initial_pred[1].item()) if self._config.run_targeted else class_id
                control[:, target] = 1

                # Terminates early if the prediction is the target.
                found_solution_func = lambda curr: curr.argmax().item() == target
                loss_target = torch.tensor([target], device=cand_i.xt.device)
            elif isinstance(self._sut, BinaryClassifierSUT):
                control = (y0 > 0).float()
                target = (1 - control[:, class_id]).item()
                control[:, class_id] = target

                # Terminates early if the sign flipped.
                found_solution_func = lambda curr: (
                    (curr[:, class_id] > 0).float().eq(target)
                ).item()

                loss_target = control
            else:
                raise NotImplementedError(
                    f"Tester does not support SUTs of type {type(self._sut)} yet."
                )

            """Here we initialize a fresh optimizer for the candidate."""
            self._optimizer.init_new(self._manipulator.hyper_net.trainable_parameters())

            cand_i.control = control
            cand_list = DiffusionCandidateList(cand_i)

            # Tracking variables for progress (the current best + budget used)
            xf_best, if_best, yf_best, budget = cand_i.xt, i0, y0, 0
            gen_data: list[dict[str, Any]] = list()
            best_fitness: dict[str, Any] = dict()
            iter_start = time()
            v_range = None
            for i in range(self._config.generations * self._config.pop_size):  # * 100 is pop size
                x_f = self._manipulator.manipulate(cand_list)
                i_f = self._manipulator.get_images(x_f)

                y_f = self._process(i_f)
                budget += i_f.size(0)

                self._objectives.evaluate_all(
                    logits=y_f,
                    initial_predictions=initial_pred,
                    images=[i0, i_f],
                    target=loss_target,
                    batch_dim=0,
                    v_range=v_range,
                    target_logit=(
                        loss_target if isinstance(self._sut, BinaryClassifierSUT) else None
                    ),
                )

                self._optimizer.assign_fitness(self._objectives.results.values())
                self._optimizer.update()
                row = {"generation": i}
                # Detach tensors to reduce memory load.
                results_detached = {
                    k: v.detach().item() if torch.is_tensor(v) else v
                    for k, v in self._objectives.results.items()
                }
                if v_range is None:
                    v_range = (0, list(results_detached.values())[-1])
                logging.info(
                    "Fitness values: " + ", ".join(f"{k}: {v}" for k, v in results_detached.items())
                )
                row |= results_detached
                gen_data.append(row)

                """Check conditions to either update best solution or terminate early."""
                if self._dominates(results_detached, best_fitness, strategy="sum"):
                    xf_best, if_best, yf_best = x_f.detach(), i_f.detach(), y_f.detach()
                    best_fitness = results_detached

                if found_solution_func(y_f):
                    logging.info(f"Found solution after {i} steps")
                    break
                del x_f, i_f, y_f
                self._cleanup()

            """Save data."""
            stats = {
                "runtime": time() - iter_start,
                "y_0": y0.cpu().detach().squeeze().tolist(),
                "y_hat": yf_best.cpu().squeeze().tolist(),
                "budget_used": budget,
            }
            log_dir = os.path.join(
                script_dir, f"runs/class_{class_id}_{self._config.save_as}_{time()}"
            )
            os.makedirs(log_dir, exist_ok=True)

            df = pd.DataFrame(gen_data)
            df.to_csv(log_dir + "/data.csv", index=False)

            self._save_tensor_as_image(i0, log_dir + f"/origin_{class_id}.png")
            self._save_tensor_as_image(if_best, log_dir + f"/taget_{target}.png")

            with open(f"{log_dir}/stats.json", "w") as f:
                f.write(json.dumps(stats))

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(k) + ': ' + str(v) for k, v in best_fitness.items()])}"
            )
            del i0, y0, cand_i, if_best, yf_best, xf_best
            self._cleanup()
            self._manipulator.make_fresh_hyper_net()  # Make a fresh hypernet for the next candidate.

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
            valid, y0 = self._sut.input_valid(image, class_id)
            if valid:
                break
            del xt, emb, image, y0
            self._cleanup()
        candidate = DiffusionCandidate(xt.squeeze(), emb, is_origin=is_origin, y=class_id)
        return candidate, y0, image

    @staticmethod
    def _dominates(curr: dict[str, Any], best: dict[str, Any], strategy: str = "pareto") -> bool:
        """
        Check if current solution dominates previous one.

        :param curr: The current solution.
        :param best: The best solution.
        :param strategy: The strategy to use (pareto, sum).
        :return: True if the current solution dominates previous one.
        :raises NotImplementedError: If strategy is not implemented.
        """
        if not best:
            return True

        curr_vals = torch.stack(
            [v.detach() if torch.is_tensor(v) else torch.tensor(v) for v in curr.values()]
        )
        best_vals = torch.stack(
            [v.detach() if torch.is_tensor(v) else torch.tensor(v) for v in best.values()]
        )
        # min all objectives
        if strategy == "pareto":
            better_or_equal: bool = (curr_vals <= best_vals).all().item()
            strictly_better: bool = (curr_vals < best_vals).any().item()
            return better_or_equal and strictly_better
        elif strategy == "sum":
            better: bool = (curr_vals.sum() < best_vals.sum()).item()
            return better
        else:
            raise NotImplementedError(f"No strategy implemented for {strategy}")
