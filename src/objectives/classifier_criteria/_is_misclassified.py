from typing import Any

from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class IsMisclassified(ClassifierCriterion):
    """Implements a criterion to check if a prediction is incorrect."""

    _name: str = "IsMisclassified"

    def __init__(self) -> None:
        """Initialize the criterion."""
        super().__init__(allow_batched=True)

    def evaluate(self, *, logits: Tensor, label_targets: list[int], **_: Any) -> list[float]:
        """
        Check if a prediction is incorrect.

        This function assumes an input range of [0, 1].

        :param logits: Tensor of predictions.
        :param label_targets: Label targets.
        :param _: Unused kwargs.
        :returns: The value.
        """
        c1 = label_targets[0]

        tensor_results: Tensor = (
            (logits.argmax(dim=1) == c1) if self._inverse else (logits.argmax(dim=1) != c1)
        )
        results: list[float] = tensor_results.float().tolist()
        return results
