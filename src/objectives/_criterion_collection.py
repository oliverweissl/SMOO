from typing import Any, Type, Union

from torch import Tensor

from ._criterion import Criterion

TCriterionResults = dict[str, Union[float, list[float], Tensor]]


class CriterionCollection:
    """A collection of criteria that simplifies calling and evaluation in addition to having better organization of results."""

    _criteria: list[Criterion]
    _results: TCriterionResults

    def __init__(self, *criteria: Criterion) -> None:
        """
        Initialize the collection of criteria.

        :param criteria: The criteria to add.
        """
        self._criteria = list(criteria)
        self._results = dict()

    def evaluate_all(self, **kwargs: Any) -> None:
        """
        Evaluate all criteria in the collection.

        :param kwargs: keyword arguments to parse.
        """
        for criterion in self._criteria:
            self._results[criterion.name] = criterion.evaluate(**kwargs)

    def precondition_all(self, iargs: dict[str, Any]) -> None:
        """
        Precondition all criteria in the collection.

        :param iargs: The input arguments.
        """
        for criterion in self._criteria:
            criterion.precondition(**iargs)

    def get_results_of(
        self, criterion: Union[Type[Criterion], Criterion]
    ) -> Union[float, list[float]]:
        """
        Get results of a specific criterion type if available.

        :param criterion: The criterion type.
        :returns: The results.
        """
        criterion = criterion if isinstance(criterion, Criterion) else criterion()
        return self._results[criterion.name]

    @property
    def results(self) -> TCriterionResults:
        """
        Get all results of the collection as a dictionary.

        :returns: The results dictionary, with keys as criterion names.
        """
        return self._results

    @property
    def names(self) -> list[str]:
        """
        Get the names of the criteria in the collection.

        :returns: The names of the criteria.
        """
        return [c.name for c in self._criteria]

    @property
    def num_objectives(self) -> int:
        """
        Get the number of objectives in the collection.

        :returns: The number of objectives in the collection.
        """
        return len(self._criteria)
