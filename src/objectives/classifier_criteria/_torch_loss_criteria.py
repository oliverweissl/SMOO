import inspect
from typing import Any

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from ._classifier_criterion import ClassifierCriterion


class TorchLossCriterion(ClassifierCriterion):
    """A wrapper to torch loss functions."""

    _name: str = "TorchLoss"

    def __init__(self, loss_fn: _Loss) -> None:
        """
        Initialize the Criterion.

        :param loss_fn: The loss function to use.
        """
        super().__init__(inverse=False, allow_batched=True)
        self._name += str(loss_fn.__class__.__name__)
        self._loss_fn = loss_fn
        self._signature = inspect.signature(loss_fn.forward)

    def evaluate(self, *, logits: Tensor, target: int, **kwargs: Any) -> Tensor:
        """
        Calculate the loss.

        :param logits: Logits tensor.
        :param target: Target class.
        :param kwargs: Other Kwargs to use.
        :returns: The value.
        """
        filtered_args = {k: v for k, v in kwargs.items() if k in self._signature.parameters}
        print(logits.shape)
        target = torch.full((logits.size(0),), target, dtype=torch.long, device=logits.device)
        return self._loss_fn(logits, target, **filtered_args)
