import inspect
from typing import Any, Optional

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

    def evaluate(self, *, logits: Tensor, **kwargs: Any) -> Tensor:
        """
        Calculate the loss.

        :param logits: Logits tensor.
        :param kwargs: Other Kwargs to use.
        :returns: The value.
        """
        filtered_args = {k: v for k, v in kwargs.items() if k in self._signature.parameters}
        result = self._loss_fn(input=logits, **filtered_args)

        v_range: Optional[tuple[float, float]] = kwargs.get("v_range")
        if v_range:
            result = (result - v_range[0]) / (v_range[1] - v_range[0])

        t = kwargs.get("target_logit")
        if t is not None:
            result = result[:, t]
        return result
