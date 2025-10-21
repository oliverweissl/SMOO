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
        apply_sigmoid: bool = True,
    ) -> None:
        """
        Initialize a binary classifier SUT.

        :param model: The model to use.
        :param batch_size: The batch size to use for prediction.
        :param device: The device to use if available.
        :param require_grad: Whether to require gradients or not.
        :param apply_sigmoid: Whether to apply sigmoid or not.
        """
        self._batch_size = batch_size
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._require_grad = require_grad
        self._apply_sigmoid = apply_sigmoid

        self._sigmoid = nn.Sigmoid()
        self._sigmoid.to(self._device)

        self._model = model
        self._model.eval()
        self._model.to(self._device)

    @SUT.standardize_inpt
    def process_input(self, inpt: Tensor) -> Tensor:
        """
        Predict class probabilities from input.

        :param inpt: Input tensor.
        :return: Predicted class probabilities on CPU.
        """
        results = []
        with torch.set_grad_enabled(self._require_grad):
            for c in inpt:
                logits = self._model(c)
                logits = self._sigmoid(logits) if self._apply_sigmoid else logits
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
