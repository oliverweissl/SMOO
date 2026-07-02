from typing import Any, Type

from .._manipulator import Manipulator
from ._perturb_candidate import PerturbCandidateList


class MultimodalManipulator(Manipulator):
    """A Manipulator that handles multi-modal inputs."""

    def __init__(self, manipulator_types: list[Type], manipulator_args: list[dict]) -> None:
        """
        Initialize the Manipulator.

        :param manipulator_types: list of modality specific manipulators to use.
        :param manipulator_args: arguments to pass to the Manipulators.
        """

        self._manipulators = [m(**k) for m, k in zip(manipulator_types, manipulator_args)]

    def manipulate(self, candidates: PerturbCandidateList, **kwargs) -> tuple[list, ...]:
        """
        Manipulate the candidates based on perturbations.

        :param candidates: candidates to manipulate.
        """
        results = [m.manipulate(candidates, **kwargs) for m in self._manipulators]
        return tuple(results)

    def synthesize(self, z: Any) -> Any:
        raise NotImplementedError("Synthesize not implemented.")
