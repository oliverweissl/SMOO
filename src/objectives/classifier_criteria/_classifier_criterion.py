from abc import abstractmethod
from typing import Any, Union

from torch import Tensor

from .._criterion import Criterion


class ClassifierCriterion(Criterion):
    """A criterion that only considers classifier outputs."""

    def __init__(self, inverse: bool = False, allow_batched: bool = False) -> None:
        """
        Initialize the ClassifierCriterion.

        :param inverse: Whether the criterion should be inverted.
        :param allow_batched: Whether the criterion supports batching.
        """
        super().__init__(inverse, allow_batched)

        # Wrap the evaluate method with logging (replaces criteria_kwargs decorator)
        if not self._allow_batched:
            original_evaluate = self.evaluate

            def logged_evaluate(
                _: Any, *, logits: Tensor, **kwargs: Any
            ) -> Union[float, list[float]]:
                return original_evaluate(logits=logits, **kwargs)

            self.evaluate = logged_evaluate.__get__(self, self.__class__)  # type: ignore[method-assign]

        if self._allow_batched:
            eval_func = self.evaluate

            def batched_evaluate(
                _: Any,
                *,
                logits: Tensor,
                batch_dim: Union[int, None] = None,
                **kwargs: Any,
            ) -> Union[float, list[float]]:
                if batch_dim is None:
                    logits = logits.unsqueeze(0)
                elif batch_dim != 0:
                    logits = logits.transpose(0, batch_dim)
                results = eval_func(logits=logits, **kwargs)

                return results[0] if batch_dim is None and isinstance(results, list) else results

            self.evaluate = batched_evaluate.__get__(self, self.__class__)  # type: ignore[method-assign]

    @abstractmethod
    def evaluate(self, *, logits: Tensor, **kwargs: Any) -> Union[float, list[float], Tensor]:
        """
        Evaluate the criterion in question.

        :param logits: The logits tensor produced by the model.
        :param kwargs: Other keyword arguments passed to the criterion.
        :returns: The value(s).
        """
        ...
