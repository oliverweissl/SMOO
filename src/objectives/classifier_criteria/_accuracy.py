from typing import Any

import torch
from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class Accuracy(ClassifierCriterion):
    """Implements Accuracy as a criterion."""

    _name: str = "Accuracy"

    def evaluate(self, *, logits: Tensor, label_targets: list[int], **_: Any) -> float:
        """
        Calculate the accuracy from prediction probabilities.

        :param logits: The predicted probabilities.
        :param label_targets: The ground truth labels.
        :param _: Unused kwargs.
        :return: The accuracy score.
        """
        y_pred_tags = torch.max(logits, dim=1)[1] if len(logits.shape) > 1 else logits

        # Convert label_targets to tensor for comparison
        label_tensor = torch.tensor(label_targets, device=logits.device, dtype=y_pred_tags.dtype)
        corr_preds = (label_tensor == y_pred_tags).float()
        acc = corr_preds.sum() / len(corr_preds)
        return float(acc.item())
