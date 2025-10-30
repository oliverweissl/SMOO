import gc
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from .manipulator import Manipulator
from .objectives import CriterionCollection
from .optimizer import Optimizer
from .sut import SUT

TEarlyTermCallable = Optional[Callable[[Any], tuple[bool, Optional[Any]]]]


class SMOO(ABC):
    """A testing object based on the SMOO-Framework."""

    """Used Components."""
    _sut: SUT
    _manipulator: Manipulator
    _optimizer: Optimizer
    _objectives: CriterionCollection

    _restrict_classes: Optional[list[int]]
    _silent: bool

    def __init__(
        self,
        *,
        sut: SUT,
        manipulator: Manipulator,
        optimizer: Optimizer,
        objectives: CriterionCollection,
        restrict_classes: Optional[list[int]],
        use_wandb: bool,
        early_termination: TEarlyTermCallable = None,
    ):
        """
        Initialize the SMOO Object.

        :param sut: The system-under-test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives used.
        :param restrict_classes: What classes to restrict predictions to.
        :param use_wandb: Whether to use wandb.
        :param early_termination: A function that can be used to early terminate the workflow.
        """

        self._sut = sut
        self._manipulator = manipulator
        self._optimizer = optimizer
        self._objectives = objectives

        self._restrict_classes = restrict_classes
        self._use_wandb = use_wandb
        self._early_termination = early_termination or (lambda _: (False, None))

    @abstractmethod
    def test(self) -> None:
        """Every workflow needs a testing loop."""
        ...

    @staticmethod
    def _cleanup() -> None:
        """Cleanup memory stuff."""
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _save_tensor_as_image(tensor: Union[NDArray, Tensor], path: str) -> None:
        """
        Save a torch tensor [0,1] as an image.

        :param tensor: The tensor to save.
        :param path: The directory to save the image to.
        """
        array = tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor
        image = array.squeeze().transpose(1, 2, 0)  # C x H x W  -> H x W x C
        image = (image * 255).astype(np.uint8)  # [0,1] -> [0, 255]
        Image.fromarray(image).save(path)

    @staticmethod
    def _get_random_seed() -> int:
        """
        A simple function to generate a seed from the os` randomness.

        :returns: A random seed.
        """
        return int.from_bytes(os.urandom(4), "little")

    @staticmethod
    def _assure_rgb(image: Tensor) -> Tensor:
        """
        Assure that an image is or can be converted to RGB.

        :param image: The image to be converted.
        :returns: The converted image (3 x H x W).
        :raises ValueError: If the image shape is not recognized.
        """
        # Add channel dimension if missing (H,W -> C,H,W)

        if image.ndim == 2:
            image = image.unsqueeze(0)

        # Channel is always at index -3 for (...,C,H,W) format
        channel_idx = image.ndim - 3
        channels = image.shape[channel_idx]

        if channels == 1:
            # Repeat single channel 3 times to make RGB
            return image.repeat_interleave(3, dim=channel_idx)
        elif channels == 3:
            return image
        else:
            raise ValueError(f"Unknown image shape. {image.shape}")

    def _process(self, x: Tensor) -> Tensor:
        """
        A wrapper to predict with additional conditions.

        :param x: The image to be predicted.
        :returns: The predicted labels.
        """
        logging.info(f"Feeding Testcases to {self._sut.__class__.__name__}.")
        y_hat = self._sut.process_input(x)
        return y_hat if self._restrict_classes is None else y_hat[:, self._restrict_classes]
