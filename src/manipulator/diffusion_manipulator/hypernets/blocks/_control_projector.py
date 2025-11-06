from functools import partial
from math import prod
from typing import Optional

import torch
from torch import Tensor, nn


class ControlProjector(nn.Module):
    """Control Projector."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        control_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize the Control Projector.

        :param input_shape: Shape of the input for the UNet2D (excluding batch_dim).
        :param control_shape: Shape of the control input (excluding batch_dim).
        :param device: Device on which to run the network.
        :param dtype: Dtype of the output.
        :raises NotImplementedError: If certain input shape is not supported yet.
        """
        super().__init__()
        # Embed the control shape into correct dimensionality for the reshape later.
        device = device or torch.get_default_device()
        dtype = dtype or torch.get_default_dtype()
        if len(control_shape) == 3:
            self.embedder = nn.Conv2d(control_shape[0], input_shape[0], 1)
        elif len(control_shape) == 2:
            embedder = nn.Conv1d(control_shape[0], input_shape[0], 1)
            embedder = embedder.to(device=device, dtype=dtype)

            resizer = nn.Linear(control_shape[1], input_shape[1] * input_shape[2])
            resizer = resizer.to(device=device, dtype=dtype)  # Ensures consistent dimensions.

            self.embedder = partial(self._1d_embedder, module=embedder, resizer=resizer)
        elif len(control_shape) == 1:
            embedder = nn.Linear(prod(control_shape), prod(input_shape))
            embedder = embedder.to(device=device, dtype=dtype)
            self.embedder = partial(self._flat_embedder, module=embedder)
        else:
            raise NotImplementedError(
                f"No behavior implemented for control shape of: {control_shape}"
            )
        self.input_shape = input_shape
        self.projector = nn.Conv2d(input_shape[0], input_shape[0], 1)

        self.to(device=device, dtype=dtype)

    def forward(self, control: Tensor) -> Tensor:
        """
        Project the control input to correct dimensions.

        :param control: Control input.
        :return: Projected control input.
        """
        x = self.embedder(control)
        return self.projector(x)

    def _1d_embedder(self, control: Tensor, module: nn.Module, resizer: nn.Module) -> Tensor:
        """
        An embedder function for 1d convolution stuff.

        :param control: Control input.
        :param module: Module to embed the input with.
        :param resizer: Resizer of sequence lengths.
        :return: Embedded input.
        """
        b = control.size(0)
        x = module(control)
        x = resizer(x)
        x = x.view(b, *self.input_shape)
        return x

    def _flat_embedder(self, control: Tensor, module: nn.Module) -> Tensor:
        """
        An embedder function for flat stuff.

        :param control: Control input.
        :param module: Module to embed the input with.
        :return: Embedded input.
        """
        b = control.size(0)
        flat = control.view(b, -1)
        x = module(flat)
        x = x.view(b, *self.input_shape)
        return x
