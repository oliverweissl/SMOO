from typing import Any, Union

import torch
from torch import Tensor

from ._image_criterion import ImageCriterion


class MatrixDistance(ImageCriterion):
    """Implements a channel-wise matrix distance measure based on torch.linalg.norm."""

    _name: str = "MatrixDistance"
    _all_norms: list[str] = ["fro", "nuc", "inf", "-inf", "1", "-1", "2", "-2"]

    def __init__(
        self, inverse: bool = False, norm: str = "fro", return_tensor: bool = False
    ) -> None:
        """
        Initialize the MatrixDistance criterion.

        :param inverse: Whether the measure should be inverted (default: False).
        :param norm: Which norm to use (default: fro).
        :param return_tensor: Whether tensor should be returned instead of list.
        """
        super().__init__(inverse, allow_batched=True)
        assert norm in self._all_norms, f"Norm {norm} not in supported norms: {self._all_norms}"
        self.norm = norm
        self._name += f"_{norm}"
        self._return_tensor = return_tensor

    def evaluate(self, *, images: list[Tensor], **_: Any) -> Union[list[float], Tensor]:
        """
        Calculate the normalized matrix distance between two tensors that range [0,1].

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: The distance.
        """
        # Expect the image tensors to have shape: B x C x H x W
        i1, i2, *_ = images  # type: ignore [assignment]
        # Upper bound of distance.
        ub = torch.linalg.matrix_norm(torch.ones_like(i1), self.norm, dim=(-2, -1))

        diff = i1 - i2
        frob = torch.linalg.matrix_norm(diff, self.norm, dim=(-2, -1))
        scaled = frob / ub

        channel_wise = scaled.mean(dim=1)
        results = torch.abs(self._inverse.real - channel_wise)
        return results if self._return_tensor else results.float().tolist()
