import logging
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ._sut import SUT
from .auxiliary_components import MonteCarloDropoutScaffold


class ClassifierSUT(SUT):
    """A multi-class classifier SUT."""

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
                c = self._ensure_image_size(c)
                logits = self._model(c)
                output = self._softmax(logits) if self._apply_softmax else logits
                results.append(output)
        res = torch.cat(results, dim=0)
        return res

    def _ensure_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt the input to a models required dimensions.

        :param x: Input tensor (Image) Shape: [B, C, H, W].
        :returns: The scaled image.
        """
        if not hasattr(self._model, "image_size"):
            return x

        target: int = self._model.image_size

        batched = x.ndim == 4

        if not batched:
            x = x.unsqueeze(0)

        _, _, h, w = x.shape

        if (h, w) == (target, target):
            return x if batched else x.squeeze(0)

        logging.warning("IMPORTANT: Image size not compatible with model - automatically resized. Please check if valid for usecase!")
        resized = F.interpolate(
            x,
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        )
        return resized if batched else resized.squeeze(0)

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing if available in API.

        :param enable: Whether to enable gradient checkpointing.
        """
        if hasattr(self._model, "gradient_checkpointing_enable") and enable:
            self._model.gradient_checkpointing_enable()
            return

        if hasattr(self._model, "gradient_checkpointing_disable") and not enable:
            self._model.gradient_checkpointing_disable()
            return

        logging.warning(
            f"Gradient checkpointing not exposed in {type(self._model)} API - gradients will be computed on whole forward pass."
        )

    def input_valid(self, inpt: Tensor, cond: int) -> tuple[bool, Tensor]:
        """
        Validate input for class membership:

        :param inpt: Input tensor.
        :param cond: The condition to check against (Class label).
        :returns: Whether the input is valid wrt. condition.
        """
        pred = self.process_input(inpt)
        if not torch.all(pred.argmax(dim=-1).eq(cond)):
            logging.info(f"Inputs not valid, got: {pred.argmax(dim=-1)}, Need: {cond}")
            return False, pred
        return True, pred
