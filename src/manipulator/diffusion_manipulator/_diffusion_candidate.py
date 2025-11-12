from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, SupportsIndex, Union, overload

import torch
from torch import Tensor

from .._candidate import Candidate, CandidateList


@dataclass
class DiffusionCandidate(Candidate):
    """A candidate solution for the diffusion based manipulator."""

    xt: Tensor  # The diffusion steps (Diffusion Steps x Image Embedding).
    class_embedding: Tensor  # The class embedding (Diffusion Steps x Class Embedding).
    is_origin: bool = False
    y: Optional[int] = None
    control: Optional[Tensor] = None  # Control Tensor for HyNeA.

    prompt: Optional[list[str]] = None  # An optional prompt for prompt-based generation.
    control_signal: Optional[Tensor] = None  # A control signal for ControlNet implementations.

    def __post_init__(self) -> None:
        """Preprocessing of some elements after initialization."""
        if isinstance(self.class_embedding, Tensor):
            self.class_embedding = torch.stack(
                [self.class_embedding.squeeze()] * len(self.xt), dim=0
            )


class DiffusionCandidateList(CandidateList[DiffusionCandidate]):
    """A list of candidate solutions for the diffusion based manipulator."""

    _candidates: list[DiffusionCandidate]
    _class_embeddings: Tensor
    _xts: Tensor  # N x  Diffusion Steps x Image Embedding

    _origin: Optional[DiffusionCandidateList] = None
    _target: Optional[DiffusionCandidateList] = None

    def __init__(
        self, *initial_candidates: DiffusionCandidate, separate_candidates: bool = True
    ) -> None:
        """
        A candidate list for the diffusion based manipulator.

        :param initial_candidates: The initial candidate solutions used for manipulation.
        :param separate_candidates: Whether the candidate list should separate target and origin candidates.
        """
        super().__init__(*initial_candidates)
        self._candidates = list(initial_candidates)

        if len(self._candidates) > 0:
            self._xts = torch.stack([c.xt for c in self._candidates], dim=0)
            self._class_embeddings = torch.stack(
                [c.class_embedding for c in self._candidates], dim=0
            )
        else:
            self._xts = None
            self._class_embeddings = None

        if separate_candidates:
            self._origin = DiffusionCandidateList(
                *(c for c in self._candidates if c.is_origin), separate_candidates=False
            )
            self._target = DiffusionCandidateList(
                *(c for c in self._candidates if not c.is_origin), separate_candidates=False
            )

    @property
    def class_embeddings(self) -> Tensor:
        """
        Get the class embeddings for the entire candidate list as a single Tensor.

        The shape of the embeddings is as follows: Num candidates x Diffusion Steps x Class Embeddings

        :return: The class embeddings for the candidates.
        """
        return self._class_embeddings

    @property
    def xts(self) -> Tensor:
        """
        Get a combined diffusion process for all candidates.

        The shape of the embeddings is as follows: Num candidates x Diffusion Steps x Latent Vectors

        :returns: The diffusion process for all candidates.
        """
        return self._xts

    @property
    def origin(self) -> Union[DiffusionCandidateList, None]:
        """
        Return origin diffusion candidates if applicable.

        :returns: The origin diffusion candidates.
        """
        return self._origin

    @property
    def target(self) -> Union[DiffusionCandidateList, None]:
        """
        Return target diffusion candidates if applicable.

        :returns: The origin diffusion candidates.
        """
        return self._target

    @overload
    def __getitem__(self, index: SupportsIndex) -> DiffusionCandidate:
        return self._candidates[index]

    @overload
    def __getitem__(self, index: slice) -> DiffusionCandidateList:
        return DiffusionCandidateList(*self._candidates[index], separate_candidates=False)

    def __getitem__(
        self, index: Union[SupportsIndex, slice]
    ) -> Union[DiffusionCandidate, DiffusionCandidateList]:
        if isinstance(index, slice):
            return DiffusionCandidateList(*self._candidates[index], separate_candidates=False)
        return self._candidates[index]

    @classmethod
    def from_diffusion_output(
        cls,
        xs: Tensor,
        emb: Tensor,
        are_origin: Optional[list[bool]] = None,
        separate_candidates: bool = True,
    ) -> DiffusionCandidateList:
        """
        Get a DiffusionCandidate list from a diffusion output.

        :param xs: The diffusion steps for the individual candidates.
        :param emb: The class embeddings for the diffusion candidates.
        :param are_origin: Set origin seeds if applicable.
        :param separate_candidates: Whether the candidate list should separate target and origin candidates.
        :returns: A DiffusionCandidate list.
        """
        num_candidates = emb.shape[0]
        if are_origin is None:
            are_origin = [False] * num_candidates
        candidates = []
        for i, o in enumerate(are_origin):
            candidate = DiffusionCandidate(class_embedding=emb[i], xt=xs[:, i, ...], is_origin=o)
            candidates.append(candidate)
        return DiffusionCandidateList(*candidates, separate_candidates=separate_candidates)
