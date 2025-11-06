from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class SUT(ABC):
    """An abstract system under test class."""

    _device: torch.device
    _dtype: torch.dtype
    _batch_size: int

    @abstractmethod
    def process_input(self, inpt: Any) -> Any:
        """
        Process the input to the SUT.

        :param inpt: The input to process.
        :return: The processed input.
        """
        ...

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing if implemented.

        :param enable: Whether to enable gradient checkpointing.
        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("This method is not implemented.")

    @abstractmethod
    def input_valid(self, inpt: Any, cond: Any) -> tuple[bool, Any]:
        """
        Validate input for a condition.

        :param inpt: The input to validate.
        :param cond: The condition to check against.
        :returns: Whether the input is valid and the output of the SUT.
        """
        ...

    @classmethod
    def standardize_inpt(cls, func: Callable) -> Callable:  # type: ignore
        def process_input_safe(self, inpt):
            if not isinstance(inpt, torch.Tensor):
                raise ValueError("Not a tensor.")
            inpt_t = inpt
            if inpt_t.device != self._device:
                inpt_t = inpt_t.to(device=self._device)
            if hasattr(self, "_dtype") and inpt_t.dtype != self._dtype:
                inpt_t = inpt_t.to(dtype=self._dtype)

            inpt_t = inpt_t.contiguous()

            batch_size = max(self._batch_size or inpt_t.size(0), 1)
            n_chunks = (inpt_t.size(0) + batch_size - 1) // batch_size
            chunks = torch.chunk(inpt_t, n_chunks, dim=0)

            assert torch.isfinite(inpt_t).all(), "input has NaNs/Infs"
            return func(self, chunks)

        return process_input_safe
