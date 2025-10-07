import logging
from abc import ABC, abstractmethod
from typing import Any, Union


class Criterion(ABC):
    """A criterion, allowing to evaluate events."""

    _name: str
    _inverse: bool
    _allow_batched: bool

    def __init__(
        self, inverse: bool = False, allow_batched: bool = False
    ) -> None:  # TODO: remove defaults here, should be set in subclass.
        """
        Initialize the criterion.

        :param inverse: Whether the criterion should be inverted.
        :param allow_batched: Whether the criterion supports batching.
        """
        self._inverse = inverse
        self._allow_batched = allow_batched
        self._name = self._name + "Inv" if inverse else self._name

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Union[float, list[float]]:
        """
        Evaluate the criterion in question.

        :param args: The arguments parsed.
        :param kwargs: The KW-Args parsed.
        :returns: The value(s).
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the criterions name.

        :returns: The name.
        """
        return self._name

    def precondition(self, **kwargs: Any) -> None:
        """
        Allows for preconditioning of the criterion.

        :param kwargs: The KW-Args parsed (should be same as the ones used in cirterion).
        """
        pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically apply wrapper if the evaluate function gets implemented.

        :param kwargs: The KW-Args parsed.
        :raises NotImplementedError: If the criterion does not support batching.
        """
        super().__init_subclass__(**kwargs)
        orig_eval = cls.evaluate
        if "batch_dim" in kwargs and not cls._allow_batched:
            raise NotImplementedError(f"Criterion {cls.__name__} does not support batching.")

        def wrapped_evaluate(self, **kwargs: Any) -> Union[float, list[float]]:
            logging.info(f"Calculating fitness with {self._name}...")
            return orig_eval(self, **kwargs)

        cls.evaluate = wrapped_evaluate  # type: ignore[method-assign]
