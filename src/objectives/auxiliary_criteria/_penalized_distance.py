from typing import Any

from torch import Tensor

from .._criterion import Criterion


class PenalizedDistance(Criterion):
    """Implements the penalized distance measure."""

    _name: str = "PenalizedDistance"
    metric: Criterion

    def __init__(self, metric: Criterion) -> None:
        """
        Initialize the Penalized Distance measure.

        :param metric: The metric used in the measure calculation.
        """
        super().__init__()
        self.metric = metric

    def evaluate(
        self, *, images: list[Tensor], logits: Tensor, initial_predictions: Tensor, **_: Any
    ) -> float:
        """
        Get penalized distance between two images using their labels.

        :param images: The images used to compute the penalized distance.
        :param logits: The logits used to compute the penalized distance.
        :param initial_predictions: The labels used to compute the penalized distance.
        :param _: Additional unused args.
        :return: The distance measure [0,1].
        """
        y1p, y2p = logits[initial_predictions[0]], logits[initial_predictions[1]]
        score = self.metric.evaluate(
            images=images, logits=logits, initial_predictions=initial_predictions
        )
        score = score[0] if isinstance(score, list) else score
        distance = (1 - score) ** (0 if y2p < y1p else 1)
        return float(distance)
