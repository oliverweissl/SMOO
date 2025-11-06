from typing import Any, Optional

from torch import Tensor

from .._criterion import Criterion


class Mean(Criterion):
    """Implements a mean-based criterion."""

    _min_max: Optional[tuple[float, float]] = None
    _name: str = "Mean"

    def __init__(self, min_max: Optional[tuple[float, float]] = None) -> None:
        """
        Initialize the Mean criterion.

        :param min_max: The min and max values for the mean to use for normalization.
        """
        super().__init__()
        self._min_max = min_max or (0, 1)

    def evaluate(
        self,
        *,
        element: Tensor,
        **_: Any,
    ) -> float:
        """
        Get the mean of the element.

        :param element: Element to mean.
        :param _: Additional unused args.
        :return: The mean.
        """
        minv, maxv = self._min_max
        result = (element.mean() - minv) / (maxv - minv)
        return float(result.item())
