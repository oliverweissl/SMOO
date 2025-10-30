from math import prod

from torch import Tensor, nn

from ._zero_conv import ZeroConv2d
from ._zero_linear import ZeroLinear


class ControlProjector(nn.Module):
    """Control Projector."""

    def __init__(self, input_shape: tuple[int, ...], control_shape: tuple[int, ...]) -> None:
        """
        Initialize the Control Projector.

        :param input_shape: Shape of the input for the UNet2D (excluding batch_dim).
        :param control_shape: Shape of the control input (excluding batch_dim).
        """
        super().__init__()
        # Embed the control shape into correct dimensionality for the reshape later.
        self.embedder = ZeroLinear(prod(control_shape), prod(input_shape))
        self.input_shape = input_shape
        self.projector = ZeroConv2d(input_shape[0])

    def forward(self, control: Tensor) -> Tensor:
        """
        Project the control input to correct dimensions.

        :param control: Control input.
        :return: Projected control input.
        """
        b = control.size(0)
        flat = control.view(b, -1)
        x = self.embedder(flat)
        x = x.view(b, *self.input_shape)
        return self.projector(x)
