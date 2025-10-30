from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .._candidate import Candidate, CandidateList


@dataclass
class MixCandidate(Candidate):
    """A simple container for candidate elements used in style mixing/ interpolation."""

    label: int  # The class label of the candidate.
    is_w0: bool = False  # Whether candidate is used for w0 calculation.
    weight: float = 1.0  # The weight of the candidate for w0 calculation.
    w_index: Optional[int] = None  # Index in the w calculation.
    w_tensor: Optional[torch.Tensor] = None  # The latent vector if already generated.
    y: Optional[int] = None  # The actual logit to consider.


class MixCandidateList(CandidateList[MixCandidate]):
    """
    A custom list like object to handle MixCandidates easily.

    Note this list object is immutable and caches getters.
    """

    _weights: list[float]
    _labels: list[int]
    _w_indices: list[int]
    _w0_candidates: Optional[MixCandidateList]
    _wn_candidates: Optional[MixCandidateList]

    def __init__(self, *initial_candidates: MixCandidate) -> None:
        """
        Initialize the MixCandidateList.

        :param initial_candidates: the initial candidates for the list.
        :raises KeyError: If the w_index is not set correctly.
        """
        super().__init__(*initial_candidates)
        """If there are elements that have no index in the original collection we assign them to ensure persistent order."""
        max_i = -1
        for i, candidate in enumerate(self.data):
            if not candidate.w_index:
                candidate.w_index = max(i, max_i + 1)
            elif candidate.w_index <= max_i:
                raise KeyError(
                    f"Something corrupted the order of this Candidate List: {self._w_indices}"
                )
            max_i = candidate.w_index

        self._weights = [elem.weight for elem in self.data]
        self._labels = [elem.label for elem in self.data]
        self._w_indices = [elem.w_index for elem in self.data if elem.w_index is not None]
        self._w_tensors = [elem.w_tensor for elem in self.data]

        self._w0_candidates = None
        self._wn_candidates = None

    @property
    def weights(self) -> list[float]:
        return self._weights

    @property
    def labels(self) -> list[int]:
        return self._labels

    @property
    def w_indices(self) -> list[int]:
        return self._w_indices

    @property
    def w_tensors(self) -> list[Optional[torch.Tensor]]:
        return self._w_tensors

    @property
    def w0_candidates(self) -> MixCandidateList:
        if not self._w0_candidates:
            self._w0_candidates = MixCandidateList(*[elem for elem in self.data if elem.is_w0])
        return self._w0_candidates

    @property
    def wn_candidates(self) -> MixCandidateList:
        if not self._wn_candidates:
            self._wn_candidates = MixCandidateList(*[elem for elem in self.data if not elem.is_w0])
        return self._wn_candidates
