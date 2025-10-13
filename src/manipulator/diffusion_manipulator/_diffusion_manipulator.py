from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor

from .._manipulator import Manipulator


class DiffusionManipulator(Manipulator, ABC):
    """An abstraction to Diffusion based manipulators."""

    @abstractmethod
    def get_diff_steps(
        self, class_labels: list[int], n_steps: int, x_0: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param class_labels: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising.
        :param x_0: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and the class embedding.
        """
        ...
