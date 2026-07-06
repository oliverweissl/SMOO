from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from src.objectives import Criterion, CriterionCollection, TCriterionResults


def get_early_termination(
    target_criterion: Criterion | Sequence[Criterion] | CriterionCollection,
    target_condition: Callable[[NDArray], NDArray],
    fulfill: str = "any",
) -> Callable[[TCriterionResults], tuple[bool, Optional[NDArray]]]:
    """
    Get an early termination condition from provided parameters.

    :param target_criterion: A criterion, multiple criteria, or a CriterionCollection.
    :param target_condition: The condition to evaluate the stacked results with.
    :param fulfill: How the condition should be fulfilled. Either "any" or "all".
    :return: The callable function.
    :raises ValueError: if fulfill is not "any" or "all".
    """
    if fulfill == "any":
        ff = np.any
    elif fulfill == "all":
        ff = np.all
    else:
        raise ValueError(f"Unknown fulfill option: {fulfill}")

    if isinstance(target_criterion, Criterion):
        criteria = [target_criterion]

    def condition_function(results: TCriterionResults) -> tuple[bool, Optional[NDArray]]:
        """
        The condition generated.

        :param results: The result dictionary.
        :returns: The condition fulfillment and the condition values.
        """
        values = []

        for criterion in criteria:
            criterion_values = results.get(criterion.name)
            if criterion_values is None:
                return False, None

            if not isinstance(criterion_values, np.ndarray):
                criterion_values = np.asarray(criterion_values)

            values.append(criterion_values)

        stacked_values = np.asarray(values)
        cond = target_condition(stacked_values)

        return bool(ff(cond)), cond

    return condition_function
