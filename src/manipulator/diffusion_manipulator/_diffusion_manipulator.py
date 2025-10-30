from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import Tensor

from .._manipulator import Manipulator


class DiffusionManipulator(Manipulator, ABC):
    """An abstraction to Diffusion based manipulators."""

    @abstractmethod
    def get_diff_steps(
        self, diff_input: Any, n_steps: Optional[int] = None, x_0: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param diff_input: The input to the manipulator.
        :param n_steps: Number of steps in the denoising.
        :param x_0: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and the class embedding.
        """
        ...
