from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .._candidate import Candidate, CandidateList


@dataclass
class MMMSample:
    """Sample Class shared by all candidates for one source example."""

    folder_path: str
    category: str
    folder_id: str
    filename: str
    clean_image_pil: Image.Image
    original_prompt: str
    target_objects: list[str]
    ground_truth_boxes: list[list[int]]
    original_size: tuple[int, int]
    clean_image_array: NDArray[np.uint8]
    baseline_iou: float | None = None
    baseline_predictions: list[dict[str, Any]] | None = None
    baseline_fail_code: str | None = None


@dataclass
class PerturbCandidate(Candidate):
    """Runtime state for one MMM optimizer individual."""

    sample: MMMSample
    prompt_template: str
    text_perturbation: list[float]
    image_pertubation: list[float]

    objects_str: str = field(init=False)
    prompt_str: str = field(init=False)
    image_array: NDArray[np.uint8] = field(init=False)

    vlm_response: str | None = None
    parsed_predictions: list[dict[str, Any]] | None = None
    matched_pred_boxes: list[list[float]] | None = None
    prompt_objects: list[str] | None = None
    objective_values: dict[str, float] = field(default_factory=dict)
    fail_code: str | None = None

    def __post_init__(self) -> None:
        """Initialize mutable candidate state from the immutable sample payload."""
        self.objects_str = ", ".join(self.sample.target_objects)
        self.prompt_str = self.prompt_template
        self.image_array = self.sample.clean_image_array.copy()

    def format_prompt(self) -> str:
        """Render the final prompt string for the current candidate state.

        :returns: Formatted prompt string.
        """
        return self.prompt_str.format(objects=self.objects_str)


class PerturbCandidateList(CandidateList[PerturbCandidate]):
    """Immutable list wrapper for perturbation candidates."""

    @property
    def image_arrays(self) -> list[NDArray[np.uint8]]:
        return [candidate.image_array for candidate in self.data]

    @property
    def prompts(self) -> list[str]:
        return [candidate.format_prompt() for candidate in self.data]

    def get_objective_values(self, key: str | list[str]) -> list[NDArray[np.float64]]:
        keys = [key] if isinstance(key, str) else key
        return [
            np.array([candidate.objective_values[name] for candidate in self.data]) for name in keys
        ]
