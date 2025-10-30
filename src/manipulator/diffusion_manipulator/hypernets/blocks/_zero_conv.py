from torch import nn


class ZeroConv2d(nn.Conv2d):
    """Zero Convolution Block."""

    def __init__(self, channels: int) -> None:
        """
        Initialize a Zero Convolution Block.

        :param channels: Number of channels in the input.
        """
        super().__init__(channels, channels, 1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
