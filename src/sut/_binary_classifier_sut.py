import logging
from typing import Optional

import torch
from torch import Tensor, nn

from ._sut import SUT


class BinaryClassifierSUT(SUT):
    """A binary classifier SUT."""

    _model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 0,
        device: Optional[torch.device] = None,
        require_grad: bool = False,
    ) -> None:
        """
        Initialize a binary classifier SUT.

        :param model: The model to use.
        :param batch_size: The batch size to use for prediction.
        :param device: The device to use if available.
        :param require_grad: Whether to require gradients or not.
        """
        self._batch_size = batch_size
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._require_grad = require_grad

        self._model = model
        self._model.eval()
        self._model.to(self._device)

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
                results.append(logits)
        res = torch.cat(results, dim=0)
        return res

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing.

        :param enable: Whether to enable gradient checkpointing.
        """
        if enable and hasattr(self._model, "gradient_checkpointing_enable"):
            self._model.gradient_checkpointing_enable()
        if not enable and hasattr(self._model, "gradient_checkpointing_disable"):
            self._model.gradient_checkpointing_disable()
        logging.warning(
            f"Toggling gradient checkpointing is not implemented for {self._model.__class__.__name__}."
        )

    def input_valid(self, inpt: Tensor, cond: int) -> tuple[bool, Tensor]:
        """
        Validate input for class membership:

        :param inpt: Input tensor.
        :param cond: The condition to check against (Class label).
        :returns: Always valid!!!.
        """
        pred = self.process_input(inpt)
        logging.warning(
            "Binary Classifier SUT always returns valid -> check if condition meets requirement."
        )
        return True, pred
