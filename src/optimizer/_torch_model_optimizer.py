import logging
from typing import Any, Callable, Iterable, NoReturn, Optional, Type

from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import LRScheduler

from ._optimizer import Optimizer


class TorchModelOptimizer(Optimizer):
    """
    An optimizer wrapper for torch models.
    This does not extend torch functionality for now, but allows for more unified usage of the Framework.
    """

    _grad_optimizer: TorchOptimizer
    _scheduler: Optional[LRScheduler]
    _loss_reductor: Callable[[tuple[Tensor, ...]], Tensor]
    _loss: Tensor

    def __init__(
        self,
        grad_optimizer: Type[TorchOptimizer],
        grad_optimizer_params: dict[str, Any],
        num_objectives: int,
        loss_reductor: Callable[[tuple[Tensor, ...]], Tensor],
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the hypernetwork optimizer.

        :param grad_optimizer: The gradient optimizer type.
        :param grad_optimizer_params: The gradient optimizer parameters.
        :param num_objectives: The number of objectives.
        :param loss_reductor: The loss reductor function that ensures we get a scalar loss per batch element.
        :param scheduler: An optional learning rate scheduler type.
        :param scheduler_params: The learning rate scheduler parameters.
        """
        super().__init__(num_objectives)
        self._grad_optimizer_t = self._optimizer_type = grad_optimizer
        self._grad_optimizer_p = grad_optimizer_params

        self._loss_reductor = loss_reductor

        self._scheduler_t = scheduler
        self._scheduler_p = scheduler_params
        if scheduler is not None:
            assert (
                scheduler_params is not None
            ), "If scheduler is provided, you must parse its arguments."

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
        if self._scheduler:
            logging.info(f"Current LR: {self._scheduler.get_last_lr()}")
            self._scheduler.step()

    def init_new(self, new_params: list[Parameter]) -> None:
        """
        Initialize new torch optimizer.

        :param new_params: The new parameters.
        """
        self.reset()
        self._grad_optimizer = self._grad_optimizer_t(new_params, **self._grad_optimizer_p)
        if self._scheduler_t:
            self._scheduler = self._scheduler_t(self._grad_optimizer, **self._scheduler_p)

    def reset(self) -> None:
        """Reset the learner to the default."""
        for name in ("_loss", "_grad_optimizer", "_scheduler"):
            if hasattr(self, name):
                delattr(self, name)

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
