"""A collection of criteria used for search based optimization methods."""

from ._archive_sparsity import ArchiveSparsity
from ._mean import Mean
from ._penalized_distance import PenalizedDistance
from ._sum import Sum

__all__ = [
    "PenalizedDistance",
    "ArchiveSparsity",
    "Sum",
    "Mean",
]
