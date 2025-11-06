from typing import Optional

from torch import nn


class ZeroConv2d(nn.Conv2d):
    """Zero 2D-Convolution Block."""

    def __init__(
        self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 1
    ) -> None:
        """
        Initialize a Zero Convolution Block.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels in the output.
        :param kernel_size: Kernel size of the convolution.
        """
        out_channels = out_channels or in_channels
        super().__init__(in_channels, out_channels, kernel_size)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ZeroConv1d(nn.Conv1d):
    """Zero 1D-Convolution Block."""

    def __init__(
        self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 1
    ) -> None:
        """
        Initialize a Zero 1D Convolution Block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels (defaults to in_channels).
        :param kernel_size: Kernel size (default: 1).
        """
        out_channels = out_channels or in_channels
        super().__init__(in_channels, out_channels, kernel_size)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
