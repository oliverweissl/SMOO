import logging
from typing import Any, Callable, Iterable, NoReturn

from torch import Tensor
from torch.optim import Optimizer as TorchOptimizer

from ._optimizer import Optimizer


class TorchModelOptimizer(Optimizer):
    """
    An optimizer wrapper for torch models.
    This does not extend torch functionality for now, but allows for more unified usage of the Framework.
    """

    _grad_optimizer: TorchOptimizer
    _loss_reductor: Callable[[tuple[Tensor, ...]], Tensor]
    _loss: Tensor

    def __init__(
        self,
        grad_optimizer: TorchOptimizer,
        num_objectives: int,
        loss_reductor: Callable[[tuple[Tensor, ...]], Tensor],
    ) -> None:
        """
        Initialize the hypernetwork optimizer.

        :param grad_optimizer: The gradient optimizer.
        :param num_objectives: The number of objectives.
        :param loss_reductor: The loss reductor function that ensures we get a scalar loss.
        """
        super().__init__(num_objectives)
        self._grad_optimizer = grad_optimizer
        self._optimizer_type = type(self._grad_optimizer)
        self._loss_reductor = loss_reductor

    def assign_fitness(self, fitness: Iterable[Tensor], *_: Any) -> None:
        """
        Overrides standard fitness assignment as we only collect loss.

        :param fitness: The loss of the current solutions.
        :param _: Unused KW-Args.
        """
        logging.info(f"Assigning fitness (loss) to {self.__class__.__name__}")
        fitness = tuple(fitness)
        assert (
            len(fitness) == self._num_objectives
        ), f"Error: {len(fitness)} Fitness (Loss) values found, {self._num_objectives} expected."

        self._loss = self._loss_reductor(fitness)

    def update(self) -> None:
        """Generate a new population based on fitness of old population."""
        self._grad_optimizer.zero_grad()
        self._loss.backward()
        self._grad_optimizer.step()

    def reset(self) -> None:
        """Reset the learner to the default."""
        del self._loss

    """Functions that are not implemented for this type of optimizer. Some overwrite default behavior tha is not applicable."""

    def get_x_current(self) -> NoReturn:
        """
        Return the current population in specific format.

        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(f"get_x_current is not implemented for {self.__class__}.")

    @property
    def best_candidates(self) -> NoReturn:
        """
        Get the best candidates so far (if more than one it is a pareto frontier).

        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(f"best_candidates is not implemented for {self.__class__}.")

    @property
    def previous_best(self) -> NoReturn:
        """
        Get the previous best candidates.

        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(f"previous_best is not implemented for {self.__class__}.")

    @property
    def n_var(self) -> NoReturn:
        """
        Get size of genome for optimizer.

        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(f"n_var is not implemented for {self.__class__}.")

    def _clip_to_bounds(self, *_) -> NoReturn:
        """
        Clip solution to bounds.

        :param _: Unused args.
        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(f"_clip_to_bounds is not implemented for {self.__class__}.")
