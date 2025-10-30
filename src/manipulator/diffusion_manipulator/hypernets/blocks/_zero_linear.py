from torch import nn


class ZeroLinear(nn.Linear):
    """A Zero-initialized Linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initializer the ZeroLinear layer.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param bias: Whether to use a bias.
        """
        super().__init__(in_features, out_features, bias)
        nn.init.zeros_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
