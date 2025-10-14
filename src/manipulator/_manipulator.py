import logging
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from ._candidate import Candidate, CandidateList


class Manipulator(ABC):
    """An abstract manipulator class."""

    @abstractmethod
    def manipulate(self, candidates: CandidateList[Candidate], **kwargs: Any) -> Any:
        """
        The manipulation function for the Manipulator.

        :param candidates: The candidates to manipulate.
        :param kwargs: Keyword arguments to pass to the manipulation function.
        :returns: The result of the manipulation.
        """
        ...

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing if implemented.

        :param enable: Whether to enable gradient checkpointing.
        """
        logging.warning(
            f"Gradient checkpointing is not implemented for {self.__class__.__name__}. Gradients will be computed for the whole forward pass."
        )

    @abstractmethod
    def get_images(self, z: Tensor) -> Tensor:
        """
        Get images from latent vector.

        :param z: The latent vector.
        :return: The decoded image, color-range [0,1] (BxCxHxW).
        """
        ...
