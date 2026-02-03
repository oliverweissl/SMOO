import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, NoReturn, Optional, Type

from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import LRScheduler

from ._optimizer import Optimizer


@dataclass(frozen=True)
class _PartialOptimizer:
    """A partial optimizer class for multi optimizer workflows."""

    optimizer_type: Type[TorchOptimizer]
    optimizer_parameters: Optional[dict[str, Any]] = None

    condition: Callable[[Parameter], bool] | None = lambda _: True

    scheduler_type: Type[LRScheduler] | None = None
    scheduler_parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """If no condition is parsed go for automatic True."""
        if self.condition is None:
            object.__setattr__(self, "condition", lambda _: True)

    def filter_model_parameters(self, parameters: list[Parameter]) -> list[Parameter]:
        """
        Select parameters to be optimized by the partial optimizer from a list of parameters.

        :param parameters: list of parameters to be filtered.
        :return: list of parameters to be optimized.
        """
        return [p for p in parameters if self.condition(p)]


class TorchModelOptimizer(Optimizer):
    """
    An optimizer wrapper for torch models.
    This does not extend torch functionality for now, but allows for more unified usage of the Framework.
    """

    _partial_optimizers: list[_PartialOptimizer]  # templates for optimizers.
    _grad_optimizers: list[TorchOptimizer]
    _schedulers: list[LRScheduler | None]
    _loss_reductor: Callable[[tuple[Tensor, ...]], Tensor]
    _loss: Tensor

    def __init__(
        self,
        grad_optimizers: Type[TorchOptimizer] | list[Type[TorchOptimizer]],
        grad_optimizer_params: dict[str, Any] | list[dict[str, Any]],
        num_objectives: int,
        loss_reductor: Callable[[tuple[Tensor, ...]], Tensor],
        optimizer_conds: list[Callable[[Tensor], bool]] | None = None,
        schedulers: Type[LRScheduler] | list[Type[LRScheduler]] | None = None,
        scheduler_params: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the hypernetwork optimizer.

        :param grad_optimizers: The gradient optimizer type(s).
        :param grad_optimizer_params: The gradient optimizer parameters.
        :param num_objectives: The number of objectives.
        :param loss_reductor: The loss reductor function that ensures we get a scalar loss per batch element.
        :param optimizer_conds: The condition to assign different optimizers to parameters.
        :param schedulers: An optional learning rate scheduler type.
        :param scheduler_params: The learning rate scheduler parameters.
        """
        super().__init__(num_objectives)
        optimizer_types = (
            grad_optimizers if isinstance(grad_optimizers, list) else [grad_optimizers]
        )
        num_optimizers = len(optimizer_types)

        """
        Here we can have 1-n optimizers that are used in the TorchModelOptimizer

        We need at least 1 optimizer parameter set to define LR etc, if there is only one parameter set,
        but multiple Optimizers, the set will be used for all of them.
        """
        if isinstance(grad_optimizer_params, list):
            assert len(grad_optimizer_params) == num_optimizers, (
                "Error: grad_optimizer_params is a list but its length does not match number of optimizers "
                f"({len(grad_optimizer_params)} != {num_optimizers})."
            )
            optimizer_param_dicts = grad_optimizer_params
        else:
            if len(optimizer_types) > 1:
                logging.warning(
                    f"Only one Optimizer Parameter set found for {num_optimizers} Optimizers, copying parameters."
                )
            optimizer_param_dicts = [dict(grad_optimizer_params) for _ in range(num_optimizers)]

        """
        To filter the model parameters for specific Optimizers we define conditions.
        If no specific condition is defined all parameters are used for each Optimizer.
        Usually you should not have this! Only use one optimizer per model parameter.
        """
        optimizer_conditions = (
            [None] * num_optimizers if optimizer_conds is None else optimizer_conds
        )
        assert (
            isinstance(optimizer_conditions, list) and len(optimizer_conditions) == num_optimizers
        ), (
            "Error: optimizer_conds must be None or a list with the same length as grad_optimizers "
            f"({len(optimizer_conditions) if isinstance(optimizer_conditions, list) else 'not a list'} != {num_optimizers})."
        )

        """
        We can have 0-n schedulers to adapt learning rate in optimizers.
        If there is no scheduler -> no adaptation.
        If there is one scheduler -> used for all Optimizers in the Class.
        If there is n -> each optimizer has its own scheduler.
        """
        scheduler_types = None
        scheduler_param_dicts = None
        if schedulers is not None:
            scheduler_types = schedulers if isinstance(schedulers, list) else [schedulers]
            if len(scheduler_types) == 1 and num_optimizers > 1:
                logging.warning(
                    f"Only one scheduler found, but {num_optimizers} optimizers; using the same scheduler for all."
                )
                scheduler_types = scheduler_types * num_optimizers
            assert len(scheduler_types) == num_optimizers, (
                "Error: schedulers must be None, a single scheduler, or a list with the same length as grad_optimizers "
                f"({len(scheduler_types)} != {num_optimizers})."
            )

            # normalize scheduler params
            assert (
                scheduler_params is not None
            ), "Error: scheduler_params must be provided when schedulers is not None."
            if isinstance(scheduler_params, list):
                assert len(scheduler_params) == num_optimizers, (
                    "Error: scheduler_params is a list but its length does not match number of optimizers "
                    f"({len(scheduler_params)} != {num_optimizers})."
                )
                scheduler_param_dicts = scheduler_params
            else:
                scheduler_param_dicts = [dict(scheduler_params) for _ in range(num_optimizers)]

            # validate params for non-None schedulers
            assert all(
                (sch is None) or (sp is not None)
                for sch, sp in zip(scheduler_types, scheduler_param_dicts)
            ), "Error: scheduler_params must be non-None for every non-None scheduler entry."

        self._optimizer_type = type(self) if num_optimizers > 1 else optimizer_types[0]
        elems = (
            zip(
                optimizer_types,
                optimizer_param_dicts,
                optimizer_conditions,
                scheduler_types,
                scheduler_param_dicts,
            )
            if scheduler_types is not None and scheduler_param_dicts is not None
            else zip(optimizer_types, optimizer_param_dicts, optimizer_conditions)
        )

        self._partial_optimizers = [_PartialOptimizer(*x) for x in elems]
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
        for optim in self._grad_optimizers:
            optim.zero_grad()
        self._loss.backward()

        for optim in self._grad_optimizers:
            optim.step()
        for scheduler in self._schedulers:
            if scheduler is not None:
                logging.info(f"Current LR: {scheduler.get_last_lr()}")
                scheduler.step()

    def init_new(self, new_params: list[Parameter]) -> None:
        """
        Initialize new torch optimizer.

        :param new_params: The new parameters.
        """
        self.reset()
        self._grad_optimizers = []
        self._schedulers = []
        for o in self._partial_optimizers:
            if not (params := o.filter_model_parameters(new_params)):
                continue
            optim = o.optimizer_type(params, **o.optimizer_parameters)
            self._grad_optimizers.append(optim)
            if o.scheduler_type is not None:
                self._schedulers.append(o.scheduler_type(optim, **o.scheduler_parameters))
            else:
                self._schedulers.append(None)

    def reset(self) -> None:
        """Reset the learner to the default."""
        for name in ("_loss", "_grad_optimizers", "_schedulers"):
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
