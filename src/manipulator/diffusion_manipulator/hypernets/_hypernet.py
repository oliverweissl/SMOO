from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class HyperNet(ABC):
    """An abstract class representing a HyperNet."""

    use_checkpoints: bool
    _device: torch.device
    _dtype: torch.dtype

    @abstractmethod
    def trainable_parameters(self) -> list[nn.Parameter]:
        """
        Parse all trainable parameters in the model.

        :returns: A list of trainable parameters in the model (Control-Layers, Zero-Layers, Control-Projector).
        """
        ...

    def _eval_module(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """
        Safely evaluate a torch module with logic to ensure checkpointing is done.

        :param module: The module to evaluate.
        :param args: The positional arguments to pass.
        :param kwargs: The additional kwargs to pass.
        :return: The output(s) of the module.
        """
        return (
            checkpoint(module, *args, **kwargs, use_reentrant=False)
            if self.use_checkpoints
            else module(*args, **kwargs)
        )

    @property
    def device(self) -> torch.device:
        """
        Get the current device of the HyperNet.

        :return: The device of the HyperNet.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """
        Get the current dtype of the HyperNet.

        :return: The current dtype of the HyperNet.
        """
        return self._dtype
