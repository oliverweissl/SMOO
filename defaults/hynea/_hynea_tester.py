import json
import logging
import os
from itertools import product
from time import time

import pandas as pd
import torch
from torch import Tensor

from src import SMOO
from src.manipulator.diffusion_manipulator import (
    DiffusionCandidate,
    DiffusionCandidateList,
    DiffusionManipulator,
)
from src.objectives import CriterionCollection
from src.optimizer import TorchModelOptimizer
from src.sut import BinaryClassifierSUT, ClassifierSUT, YoloSUT

from ._experiment_config import ExperimentConfig


class HyNeATester(SMOO):
    """A tester class that implements the HyNeA method."""

    _manipulator: DiffusionManipulator
    _optimizer: TorchModelOptimizer
    _sut: ClassifierSUT
    _config: ExperimentConfig

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: DiffusionManipulator,
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
            initial_pred = torch.argsort(y0[0], descending=True)
            control = torch.zeros(1, *self._manipulator._control_shape, device=cand_i.xt.device)
            assert (
                control.shape == y0.shape
            ), f"Error in Control shape. Got {control.shape} instead of {y0.shape}."

            # Adapt the functionality of testing based on the SUT.
            if isinstance(self._sut, ClassifierSUT):
                target = int(initial_pred[1].item()) if self._config.run_targeted else class_id
                control[:, target] = 1

                # Updates solution if the prediction of target increases.
                update_solution_func = lambda curr, best: curr[:target] > best[:target]
                # Terminates early if the prediction is the target.
                found_solution_func = lambda curr: curr.argmax().item() == target
            elif isinstance(self._sut, BinaryClassifierSUT):
                curr_pred = (y0[0][class_id] > 0).real
                target = 1 - curr_pred
                control[:, class_id] = target

                # Updates solution if it wanders closer to a sign flip.
                update_solution_func = lambda curr, best: curr[:class_id] < best[:class_id]
                # Terminates early if the sign flipped.
                found_solution_func = lambda curr: torch.all(
                    (((curr[:class_id] / torch.abs(curr[:class_id])) + 1) / 2).eq(target)
                )
            else:
                raise NotImplementedError(
                    f"Tester does not support SUTs of type {type(self._sut)} yet."
                )
            cand_i.control = control
            cand_list = DiffusionCandidateList(cand_i)

            xf_best, if_best, yf_best, budget = cand_i.xt, i0, y0, 0
            gen_data, best_fitness = [], {}
            iter_start = time()
            for i in range(self._config.generations * self._config.pop_size):  # * 100 is pop size
                x_f = self._manipulator.manipulate(cand_list)
                i_f = self._manipulator.get_images(x_f)

                y_f = self._process(i_f)
                budget += i_f.size(0)

                self._objectives.evaluate_all(
                    logits=y_f,
                    initial_predictions=initial_pred,
                    images=[i0, i_f],
                    batch_dim=0,
                )

                row = {"generation": i}
                row |= self._objectives.results
                gen_data.append(row)
                self._optimizer.assign_fitness(self._objectives.results.values())

                """Check conditions to either update best solution or terminate early."""
                if update_solution_func(y_f, yf_best):
                    xf_best, if_best, yf_best = x_f, i_f, y_f
                    best_fitness = self._objectives.results

                if found_solution_func(y_f):
                    logging.info(f"Found solution after {i} steps")
                    break

            """Save data."""
            stats = {
                "runtime": time() - iter_start,
                "y_0": y0.cpu().detach().squeeze().tolist(),
                "y_hat": yf_best.cpu().detach().squeeze().tolist(),
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

            del x_f, i_f, y_f, i0, y0, cand_i, if_best, yf_best, xf_best
            torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()
        candidate = DiffusionCandidate(xt.squeeze(), emb, is_origin=is_origin, y=class_id)
        return candidate, y0, image
