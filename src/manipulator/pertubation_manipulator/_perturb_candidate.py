from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .._candidate import Candidate, CandidateList


@dataclass
class PerturbCandidate(Candidate):
    """A simple container for candidate elements used in pertubation."""

    prompt: str
    objects: list[str]
    image: str

    original_bboxes: list[list[int]]

    text_perturbation: list[float]  # Genome
    image_pertubation: list[float]  # Genome

    objects_str: str = field(init=False)
    prompt_str: str = field(init=False)
    image_array: np.ndarray

    def __post_init__(self) -> None:
        """Post init processing of stuff."""
        self.objects_str = ", ".join(self.objects)
        self.prompt_str = self.prompt
        img = Image.open(self.image).convert("RGB")
        self.image_array = np.array(img)

    def format_prompt(self) -> str:
        """
        Format the final prompt.

        :return: Formatted prompt.
        """
        return self.prompt_str.format(objects=self.objects_str)


class PerturbCandidateList(CandidateList[PerturbCandidate]):
    """
    A custom list like object to handle PerturbCandidate easily.

    Note this list object is immutable and caches getters.
    """
