from typing import Any, Optional

import torch
from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class AdversarialDistance(ClassifierCriterion):
    """An objective function that allows to optimize for adversarial examples."""

    _name: str = "AdvD"

    def __init__(
        self,
        target_pair: bool = False,
        inverse: bool = False,
        exp_decay_lambda: Optional[float] = None,
    ) -> None:
        """
        Initialize the AdversarialDistance objective.

        :param target_pair: If true, we want to target the adversarial examples of specific class pairs (class X -> class Y) else (class X -> any class).
        :param inverse: Whether the measure should be inverted.
        :param exp_decay_lambda: Factor for exponential decay f(x) = e^(-lambda*x).
        :raises NotImplementedError: If inverse of funcion is required.
        """
        super().__init__(inverse=inverse, allow_batched=True)
        self._target_pair = target_pair
        self._exp_decay_lambda = exp_decay_lambda
        if self._inverse:
            raise NotImplementedError("Inverse does not function properly yet.")

    def evaluate(self, *, logits: Tensor, label_targets: list[int], **_: Any) -> list[float]:
        """
        Calculate the confidence balance of 2 confidence values.

        This function assumes an input range of [0, 1].

        :param logits: Logits tensor.
        :param label_targets: Label targets used to determine targets of balance.
        :param _: Unused kwargs.
        :returns: The value in range [0,1].
        """
        origin, target, *_ = label_targets  # type: ignore [assignment]

        if self._target_pair:
            second_term = logits[target].item()
        else:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, origin] = True
            second_term = logits.masked_fill(mask, -torch.inf).max(dim=1).values

        partial = (-1) ** (2 - self._inverse.real) * (
            logits[:, origin] - second_term
        ) + self._inverse.real

        if self._exp_decay_lambda is not None:
            """
            Here we apply the exponential decay to balance the effect of the distance measure.
            For adversarial testing we are not interested in flipping the prediction probabilities, rather approach a failure case.
            If the decay is linear, this measure can easily overpower the image distance metrics.
            This would result in perfect misclassifications, but with rather lager image distances.
            """
            partial = torch.exp(-self._exp_decay_lambda * partial)
        results: list[float] = partial.tolist()
        return results
