from typing import Any, Optional

import torch
from torch import Tensor
from ultralytics import YOLO

from ._sut import SUT


class YoloSUT(SUT):
    """A YOLO based SUT."""

    _model: YOLO
    _batch_size: int

    def __init__(
        self,
        model: YOLO,
        batch_size: int = 0,
        device: Optional[torch.device] = None,
        return_confidences: bool = False,
        return_bboxes: bool = False,
        objectness_exists: bool = True,
        return_objectness: bool = False,
        return_indices: Optional[list[int]] = None,
    ) -> None:
        """
        Initialize the classifier SUT.

        :param model: The model to use.
        :param batch_size: The batch size to use for prediction.
        :param device: The device to use if available.
        :param return_bboxes: Whether to return bboxes.
        :param return_objectness: Whether to return objectness.
        :param return_confidences: Whether to return the confidence of predictions.
        :param objectness_exists: Whether the model outputs objectness.
        :param return_indices: Indices of the outputs to return (default return all).
        """
        self._batch_size = batch_size
        self._model = model.model
        self._model.to(str(device))

        self._objectness_exists = objectness_exists

        self._return_confidences = return_confidences
        self._return_bbox = return_bboxes
        self._return_objectness = return_objectness

        self._return_indices = return_indices

        assert (
            return_confidences or return_bboxes or return_objectness
        ), "At least one of the outputs must be returned."
        assert (
            sum((return_confidences.real, return_bboxes.real, return_objectness.real)) == 1
        ), "For now only returning one output is supported."
        assert (
            not return_objectness or objectness_exists
        ), "Objectness exist if it should be returned."

    def process_input(self, inpt: Tensor) -> Tensor:
        """
        Predict class probabilities from input.

        :param inpt: Input tensor.
        :return: Predicted class probabilities on CPU.
        :raises ValueError: If no output is selected.
        :raises ValueError: If objectness is expected but not computed.
        """
        batch_size = max(
            self._batch_size or inpt.size(0), 1
        )  # If batchsize == 0 -> do the whole input.
        n_chunks = (inpt.size(0) + batch_size - 1) // batch_size
        chunks = torch.chunk(inpt, n_chunks, dim=0)

        assert torch.isfinite(inpt).all(), "input has NaNs/Infs"

        objectness, confidences, bboxes = [], [], []
        with torch.no_grad():
            for c in chunks:
                output = self._model(c)[0]
                bboxes.append(output[:, :4, :])
                if self._objectness_exists:
                    objectness.append(output[:, 4, :])
                idx_start = 5 if self._objectness_exists else 4
                confidences.append(output[:, idx_start:, :])

        objectness = torch.cat(objectness, dim=0)
        confidences = torch.cat(confidences, dim=0)
        bboxes = torch.cat(bboxes, dim=0)

        """Here we sort all predictions in the image by the magnitude of confidences."""
        top_score, _ = confidences.max(dim=1)
        sort = top_score.argsort(dim=1, descending=True)
        if self._return_confidences:
            data = confidences
        elif self._return_bbox:
            data = bboxes
        elif self._return_objectness:
            if objectness.numel() > 0:
                data = objectness.unsqueeze(1)
            else:
                raise ValueError("Objectness expected but not computed.")
        else:
            raise ValueError("No output selected.")
        sorted_data = torch.gather(
            data, dim=2, index=sort.unsqueeze(1).expand(-1, data.size(1), -1)
        )

        """We select which indices to return, if we only return a singular datapoint, we remove the datapoint dimension."""
        return_indices = self._return_indices or torch.arange(sorted_data.shape[-1])
        return_data = sorted_data[:, :, return_indices]
        return return_data if len(return_indices) > 1 else return_data.squeeze(-1)

    def input_valid(self, inpt: Any, cond: Any) -> tuple[bool, Any]:
        """
        Validate input for YOLO:

        :param inpt: Input tensor.
        :param cond: The condition to check against.
        :raises NotImplementedError: If not implemented.
        """
        raise NotImplementedError("input_valid not implemented for this function.")
