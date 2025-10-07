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
    SitHyNeAManipulator,
)
from src.objectives import CriterionCollection
from src.optimizer import TorchModelOptimizer
from src.sut import ClassifierSUT

from ._experiment_config import ExperimentConfig


class HyNeATester(SMOO):
    """A tester class that implements the HyNeA method."""

    _manipulator: SitHyNeAManipulator
    _optimizer: TorchModelOptimizer
    _sut: ClassifierSUT
    _config: ExperimentConfig

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: SitHyNeAManipulator,
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
        # TODO: fix checkpointing for better memory usage
        self._sut.gradient_checkpointing(enable=True)
        self._manipulator.gradient_checkpointing(enable=True)

        self._config = config

    def test(self) -> None:
        """Start the HyNeA-based testing."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for class_id, sample_idx in product(
            self._config.classes, range(self._config.samples_per_class)
        ):
            logging.info(f"Test class {class_id}, sample idx {sample_idx}.")
            cand_i, y0, i0 = self._find_valid_candidate(class_id, is_origin=True)
            _, second, *_ = torch.argsort(y0[0], descending=True)

            target_class = int(second.item()) if self._config.run_targeted else class_id
            control = torch.zeros(1, 1000, device=self._manipulator._device)
            control[:, target_class] = 1

            xf_best, if_best, yf_best, budget = cand_i.xt, i0, y0, 0
            gen_data, best_fitness = [], {}
            iter_start = time()
            for i in range(self._config.generations * self._config.pop_size):  # * 100 is pop size
                x_f = self._manipulator.manipulate(
                    x=cand_i.xt[0].unsqueeze(0), y=[class_id], c=control
                )
                i_f = self._manipulator.get_image(x_f)

                y_f = self._process(i_f)
                budget += i_f.size(0)

                # We need to unsqueeze the target to match batch size.
                self._objectives.evaluate_all(
                    logits=y_f,
                    target=class_id,
                    images=[i0, i_f],
                    batch_dim=0,
                )

                row = {"generation": i}
                row |= self._objectives.results
                gen_data.append(row)
                self._optimizer.assign_fitness(self._objectives.results.values())
                if y_f[:, target_class] > yf_best[:, target_class]:
                    xf_best, if_best, yf_best = x_f, i_f, y_f
                    best_fitness = self._objectives.results

                if y_f.argmax().item() == target_class:
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
            self._save_tensor_as_image(if_best, log_dir + f"/taget_{target_class}.png")

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
            image = self._manipulator.get_image(xt[-1])
            y_hat = self._process(image)
            if torch.argmax(y_hat) == class_id:
                break
            logging.warning(
                f"Failed to find candidate for {class_id}, predicted {torch.argmax(y_hat)}"
            )
            del xt, emb, image, y_hat
            torch.cuda.empty_cache()
        candidate = DiffusionCandidate(xt.squeeze(), emb, is_origin=is_origin)
        return candidate, y_hat, image
