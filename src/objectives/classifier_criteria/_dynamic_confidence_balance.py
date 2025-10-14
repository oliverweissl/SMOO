from typing import Any, Optional

import torch
from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class DynamicConfidenceBalance(ClassifierCriterion):
    """Implements a dynamic confidence balance measure."""

    _name: str = "DynCB"
    _target_primary: Optional[bool]

    def __init__(self, inverse: bool = False, target_primary: Optional[bool] = None) -> None:
        """
        Initialize the criterion.

        :param inverse: Whether the measure should be inverted.
        :param target_primary: Whether y1 is the focus of the measure or yp is. If none, neither is in focus.
        """
        super().__init__(inverse=inverse, allow_batched=True)
        self._target_primary = target_primary

    def evaluate(self, *, logits: Tensor, initial_predictions: Tensor, **_: Any) -> list[float]:
        """
        Calculate the confidence balance of 2 confidence values.

        This function assumes an input range of [0, 1].

        :param logits: Logits tensor.
        :param initial_predictions: The initial predictions for the search as labels.
        :param _: Unused kwargs.
        :returns: The value.
        """
        origin = initial_predictions[0]  # The primary class

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, origin] = True
        second_term = logits.masked_fill(mask, float("-inf")).max(dim=1).values

        s = logits[:, origin] + second_term
        d = torch.abs(logits[:, origin] - second_term)

        target = 0
        if self._target_primary is not None:
            if self._target_primary:
                target = second_term
            elif not self._target_primary:
                target = logits[:, origin]

        result: list[float] = torch.abs(self._inverse.real - target - d / s).float().tolist()
        return result
