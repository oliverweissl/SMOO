from typing import Optional

import torch
from torch import Tensor, nn

from ._sut import SUT
from .auxiliary_components import MonteCarloDropoutScaffold


class ClassifierSUT(SUT):
    """A classifier SUT."""

    _model: nn.Module
    _softmax: nn.Softmax

    _apply_softmax: bool
    _batch_size: int

    def __init__(
        self,
        model: nn.Module,
        apply_softmax: bool = False,
        use_mcd: bool = False,
        batch_size: int = 0,
        device: Optional[torch.device] = None,
        require_grad: bool = False,
    ) -> None:
        """
        Initialize the classifier SUT.

        :param model: The model to use.
        :param apply_softmax: Whether to apply softmax or not.
        :param use_mcd: Whether to use Monte Carlo Dropout or not.
        :param batch_size: The batch size to use for prediction.
        :param device: The device to use if available.
        :param require_grad: Whether to require gradients or not.
        """
        self._apply_softmax = apply_softmax
        self._batch_size = batch_size
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = MonteCarloDropoutScaffold(model) if use_mcd else model
        self._model.eval()
        self._softmax = nn.Softmax(dim=-1)

        self._require_grad = require_grad
        self._model.to(self._device)
        self._softmax.to(self._device)

    def process_input(self, inpt: Tensor) -> Tensor:
        """
        Predict class probabilities from input.

        :param inpt: Input tensor.
        :return: Predicted class probabilities on CPU.
        """
        if inpt.device != self._device:
            inpt = inpt.to(self._device)

        batch_size = max(
            self._batch_size or inpt.size(0), 1
        )  # If batchsize == 0 -> do whole input.
        n_chunks = (inpt.size(0) + batch_size - 1) // batch_size
        chunks = torch.chunk(inpt, n_chunks, dim=0)

        assert torch.isfinite(inpt).all(), "input has NaNs/Infs"

        results = []
        with torch.set_grad_enabled(self._require_grad):
            for c in chunks:
                logits = self._model(c)
                output = self._softmax(logits) if self._apply_softmax else logits
                results.append(output)
        res = torch.cat(results, dim=0)
        return res

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing.

        :param enable: Whether to enable gradient checkpointing.
        """
        if enable:
            self._model.gradient_checkpointing_enable()
        else:
            self._model.gradient_checkpointing_disable()
