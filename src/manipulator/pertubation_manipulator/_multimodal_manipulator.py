from typing import Any

from .._manipulator import Manipulator
from ._perturb_candidate import PerturbCandidateList


class MultimodalManipulator(Manipulator):
    """A Manipulator that handles multi-modal inputs."""

    def __init__(
        self, manipulator_types: list[type[Manipulator]], manipulator_args: list[dict[str, Any]]
    ) -> None:
        """
        Initialize the Manipulator.

        :param manipulator_types: list of modality specific manipulators to use.
        :param manipulator_args: arguments to pass to the Manipulators.
        """
        self._manipulators = [m(**k) for m, k in zip(manipulator_types, manipulator_args)]

    def image_dim(self) -> int:
        """Return the total image perturbation dimensionality.

        :returns: Number of image perturbation parameters.
        """
        return sum(m.image_dim() for m in self._manipulators if hasattr(m, "image_dim"))

    def text_dim(self) -> int:
        """Return the total text perturbation dimensionality.

        :returns: Number of text perturbation parameters.
        """
        return sum(m.text_dim() for m in self._manipulators if hasattr(m, "text_dim"))

    def manipulate(self, candidates: PerturbCandidateList, **kwargs: Any) -> PerturbCandidateList:
        """
        Manipulate the candidates based on perturbations.

        :param candidates: candidates to manipulate.
        :param kwargs: Keyword arguments forwarded to inner manipulators.
        :returns: Per-manipulator results.
        """
        for m in self._manipulators:
            candidates = m.manipulate(candidates)
        return candidates

    def synthesize(self, z: Any) -> Any:
        raise NotImplementedError("Synthesize not implemented.")
