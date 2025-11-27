import glob
import itertools
import json
import logging
import os
import random
from itertools import product
from time import time
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from src import SMOO
from src.manipulator.diffusion_manipulator import (
    DiffusionCandidate,
    DiffusionCandidateList,
    LDMHyNeAManipulator,
    SDCNHyNeAManipulator,
    SitHyNeAManipulator,
)
from src.objectives import CriterionCollection
from src.optimizer import TorchModelOptimizer
from src.sut import BinaryClassifierSUT, ClassifierSUT, YoloSUT
from src.utils.exceptions import ExceededIterationBudget

from ._experiment_config import ExperimentConfig


class HyNeATester(SMOO):
    """A tester class that implements the HyNeA method."""

    _manipulator: LDMHyNeAManipulator | SitHyNeAManipulator | SDCNHyNeAManipulator
    _optimizer: TorchModelOptimizer
    _sut: ClassifierSUT | BinaryClassifierSUT | YoloSUT
    _config: ExperimentConfig

    def __init__(
        self,
        *,
        sut: ClassifierSUT | BinaryClassifierSUT | YoloSUT,
        manipulator: LDMHyNeAManipulator | SitHyNeAManipulator | SDCNHyNeAManipulator,
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

            try:
                cand_i, y0, i0 = self._find_valid_candidate(
                    [class_id], is_origin=True, class_id=class_id
                )
            except ExceededIterationBudget:
                logging.warning("Budget for candidate generation exceeded. Skipping.")
                continue  # Move on to the next iteration

            initial_pred = y0[0]  # [1, X] -> [X]
            control = torch.zeros(1, *self._manipulator.control_shape, device=cand_i.xt.device)
            assert (
                control.shape == y0.shape
            ), f"Error in Control shape. Got {control.shape} instead of {y0.shape}."

            if isinstance(self._sut, ClassifierSUT):
                """
                When using a Multi-Class classification system as our SUT, we do the following:
                - Sort the initial prediction to find second most likely class
                - Define target based, if targeted testing is wanted.
                - Create a control signal to reflect the target behavior of the SUT.
                - Define a early termination function that checks whether we predict the target.
                - Define a target for the torch loss function to work properly.
                """
                initial_pred = torch.argsort(
                    initial_pred, descending=True
                )  # Only used in multi-class classification
                target = int(initial_pred[1].item()) if self._config.run_targeted else class_id
                control[:, target] = 1

                found_solution_func = lambda curr: curr.argmax().item() == target
                loss_target = torch.tensor([target], device=cand_i.xt.device)
            elif isinstance(self._sut, BinaryClassifierSUT):
                """
                When using a Binary Classifier as our SUT, we do the following:
                - Define our target as opposing predictions than in the initial prediction, for the logits we target.
                - Create a control signal that reflect this sign-flip behavior (operate on logits).
                - Define a early termination function, that checks whether the sign has flipped.
                - Define the target for the torch loss function, which is equal to the control signal.
                """
                control = (y0 > 0).float()
                target = (1 - control[:, class_id]).item()
                control[:, class_id] = target

                found_solution_func = lambda curr: (
                    (curr[:, class_id] > 0).float().eq(target)
                ).any()

                loss_target = control
            elif isinstance(self._sut, YoloSUT):
                """
                When using a YOLO Object detector as our SUT, we do the following:
                - Define the target as the second most likely initial predictions for all detections if they do not correspond to our testing class.
                - Create a control signal that enforces different prediction for all detections.
                - Define a early termination function, that checks whether all detections have a confidence of less than 0.5 for the class we test.
                - Define the target for the torch loss function, which is used for BCE-Loss as in YOLO training.
                """
                initial_pred = torch.argsort(initial_pred, dim=0, descending=True)  # [80, N]
                target = initial_pred[1, :]  # [N]

                mask = target == class_id
                target[mask] = initial_pred[0, :][mask]

                control[:, target, :] = 1

                found_solution_func = lambda curr: (curr.argmax(dim=1) != class_id).all().item()
                loss_target = target.clone().detach()[:5]
                target = target[0]  # To make file creation less chaotic
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
                if isinstance(self._sut, YoloSUT):
                    y_f = (
                        y_f.squeeze().T
                    )  # Yolo gives [1, 80, Detections] -> reshape to [Detections, 80] for the loss_target to fit.
                    y_f_eval = y_f[:5]  # Only take top-k to keep gradients relevant
                else:
                    y_f_eval = y_f

                budget += i_f.size(0)

                self._objectives.evaluate_all(
                    logits=y_f_eval,
                    initial_predictions=initial_pred,
                    images=[i0, i_f],
                    target=loss_target,
                    batch_dim=0,
                    v_range=v_range,
                    target_logit=class_id if isinstance(self._sut, BinaryClassifierSUT) else None,
                )

                self._optimizer.assign_fitness(self._objectives.results.values())
                self._optimizer.update()
                row = {"generation": i}

                results_detached = {
                    k: v.detach().item() if torch.is_tensor(v) else v
                    for k, v in self._objectives.results.items()
                }
                if v_range is None and not isinstance(self._sut, YoloSUT):
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

                if found_solution_func(y_f_eval):
                    xf_best, if_best, yf_best = x_f.detach(), i_f.detach(), y_f.detach()
                    best_fitness = results_detached
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

            log_dir = os.path.join(script_dir, f"{self._config.save_as}/class_{class_id}_{time()}")
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
        self,
        diff_input: Any,
        class_id: int,
        is_origin: bool = False,
        max_iterations: int = 100,
    ) -> tuple[DiffusionCandidate, Tensor, Tensor]:
        """
        Sample valid candidates. If the SUT is YoloSUT, automatically cycle
        through random control-signals and prompts until one yields a valid image.

        :param diff_input: The diffusion inputs passed from main loop.
        :param class_id: The current class ID.
        :param is_origin: Whether the candidate is an origin candidate.
        :param max_iterations: The maximum number of iterations to find a candidate.
        :return: The found Candidate, the initial prediction, and the corresponding image.
        :raises ExceededIterationBudget: If the budget for finding candidate is exceeded.
        """
        n_iter = 0

        # Define generator for YoloSUT control signals
        if isinstance(self._sut, YoloSUT):
            yolo_prompt = (
                "A photorealistic urban traffic scene with cars, traffic lights, and stop signs, "
                f"clear skies, daytime, featuring a {self._sut.class_mapping.get(class_id)}"
            )

            def _control_signal_generator():
                files = glob.glob("_data_semantics/training/semantic_rgb/*.png")
                while True:
                    random_file = random.choice(files)
                    control_img = Image.open(random_file)
                    width, height = control_img.size
                    target_height = 512
                    scale_factor = target_height / height
                    new_width = int(width * scale_factor)
                    control_img = control_img.resize((new_width, target_height), Image.LANCZOS)
                    left = (new_width - target_height) // 2
                    right = left + target_height
                    control_img = control_img.crop((left, 0, right, 512))
                    control_signal = ToTensor()(control_img).unsqueeze(0)
                    yield (control_signal, [yolo_prompt])

            input_cycle = _control_signal_generator()
        else:
            input_cycle = itertools.cycle([diff_input])

        while True:
            current_input = next(input_cycle)

            xt, emb = self._manipulator.get_diff_steps(current_input)
            image = self._manipulator.get_images(xt[-1])
            valid, y0 = self._sut.input_valid(image, class_id)

            if valid:
                candidate = DiffusionCandidate(xt.squeeze(), emb, is_origin=is_origin, y=class_id)
                if isinstance(self._sut, YoloSUT):
                    candidate.control_signal, candidate.prompt = current_input
                return candidate, y0, image

            del xt, emb, image, y0
            self._cleanup()
            n_iter += 1

            if n_iter >= max_iterations > 0:
                raise ExceededIterationBudget(
                    "Maximum number of iterations reached in candidate generation."
                )

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
