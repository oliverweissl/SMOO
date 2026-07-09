import logging
from typing import Any, Optional, Type

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from ._optimizer import Optimizer
from .auxiliary_components import OptimizerCandidate


class PymooOptimizer(Optimizer):
    """A Learner class for easy Pymoo integration"""

    _pymoo_algo: GeneticAlgorithm
    _problem: Problem
    _pop_current: Population
    _bounds: tuple[int, int]
    _shape: tuple[int, ...]

    _params: dict[str, Any]
    _algorithm: Type[GeneticAlgorithm]

    def __init__(
        self,
        bounds: tuple[int, int],
        algorithm: Type[GeneticAlgorithm],
        algo_params: dict[str, Any],
        num_objectives: int,
        solution_shape: tuple[int, ...],
        save_history: bool = True,
    ) -> None:
        """
        Initialize the genetic learner.

        :param bounds: Bounds for the optimizer.
        :param algorithm: The pymoo Algorithm.
        :param algo_params: Parameters for the pymoo Algorithm.
        :param num_objectives: The number of objectives the learner can handle.
        :param solution_shape: The shape of the solution arrays.
        :param save_history: Whether pymoo should retain full generation history.
        """
        super().__init__(num_objectives)
        self._params = algo_params
        self._algorithm = algorithm
        self._bounds = bounds
        self._save_history = save_history
        self._evaluator = Evaluator()

        self.update_problem(solution_shape)
        self._optimizer_type = type(self._pymoo_algo)

    def update(self) -> None:
        """
        Advance pymoo by one generation.
        """
        logging.info("Advancing pymoo generation...")
        static = StaticProblem(self._problem, F=np.column_stack(self._fitness))
        self._evaluator.eval(static, self._pop_current)
        self._pymoo_algo.tell(self._pop_current)

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._clip_to_bounds(self._pop_current.get("X"))

    def get_x_current(self) -> NDArray:
        """
        Return the current population in a specific format.

        :return: The currently best genome.
        """
        return self._x_current.reshape((self._x_current.shape[0], *self._shape))

    def update_problem(
        self, solution_shape: tuple[int, ...], sampling: Optional[NDArray] = None
    ) -> None:
        """
        Change problem shape of optimization.

        :param solution_shape: The new solution shape.
        :param sampling: An initial solution array if available.
        """
        if sampling is not None:
            assert np.prod(solution_shape) == np.prod(
                sampling.shape[1:]
            ), f"ERROR: sampling shape {sampling.shape[1:]}, does not conform to solution size {solution_shape}."
            x0 = sampling.reshape(sampling.shape[0], -1)
            x0 = self._clip_to_bounds(x0)
            self._params["sampling"] = x0

        self._shape = solution_shape
        self._n_var = int(np.prod(solution_shape))
        self._pymoo_algo = self._algorithm(**self._params, save_history=self._save_history)

        self._problem = Problem(
            n_var=self._n_var,
            n_obj=self._num_objectives,
            xl=self._bounds[0],
            xu=self._bounds[1],
            vtype=float,
        )
        self._pymoo_algo.setup(self._problem, termination=NoTermination())
        self.reset()

    def reset(self) -> None:
        """Resets the optimizer."""
        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._clip_to_bounds(self._pop_current.get("X"))

        self._best_candidates = [
            OptimizerCandidate(
                solution=np.random.uniform(
                    high=self._bounds[0], low=self._bounds[1], size=self._n_var
                ),
                fitness=[np.inf] * self._num_objectives,
            )
        ]
        self._previous_best = self._best_candidates.copy()

    @property
    def best_solutions_reshaped(self) -> list[NDArray]:
        """
        Get the best solutions in correct shape.

        :return: The solutions.
        """
        return [
            c.solution.reshape(self._shape) for c in self._best_candidates if c.solution is not None
        ]
