import logging
from typing import Any, Optional

import torch
from torch import Tensor
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.model import Model

from ._sut import SUT


class YoloSUT(SUT):
    """A YOLO based SUT."""

    _model: Model
    _batch_size: int
    _filter_classes: Optional[list[int]]
    _nms_threshold: float

    def __init__(
        self,
        model: YOLO | str,
        batch_size: int = 0,
        device: Optional[torch.device] = None,
        return_confidences: bool = False,
        return_bboxes: bool = False,
        objectness_exists: bool = True,
        return_objectness: bool = False,
        return_indices: Optional[list[int]] = None,
        require_grad: bool = False,
        filter_classes: Optional[int | list[int]] = None,
        nms_threshold: float = 0.0,
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
        :param require_grad: Whether to require grad.
        :param filter_classes: The class to filter predictions for (only output if class is most likely pred).
        :param nms_threshold: A threshold for non-maximum suppression (overlapping bbox are merged, default = 0.0 -> no merging)
        """
        self._batch_size = batch_size
        self._model = model.model if isinstance(model, YOLO) else YOLO(model).model
        self._model.to(device)

        self._objectness_exists = objectness_exists

        self._return_confidences = return_confidences
        self._return_bbox = return_bboxes
        self._return_objectness = return_objectness

        self._return_indices = return_indices

        self._require_grad = require_grad
        self._device = device
        self._dtype = next(self._model.parameters()).dtype

        self._filter_classes = (
            filter_classes
            if isinstance(filter_classes, list) or filter_classes is None
            else [filter_classes]
        )
        assert 1.0 >= nms_threshold >= 0.0, "Error: nms-threshold must be in [0., 1.]"
        self._nms_threshold = nms_threshold
        assert (
            return_confidences or return_bboxes or return_objectness
        ), "At least one of the outputs must be returned."
        assert (
            not return_objectness or objectness_exists
        ), "Objectness exist if it should be returned."

    @SUT.standardize_inpt
    def process_input(self, inpt: Tensor) -> Tensor:
        """
        Predict class probabilities from input.

        :param inpt: Input tensor.
        :return: Predicted elements in order [Conf, BBOX, Objectness].
        :raises ValueError: If no output is selected.
        :raises ValueError: If objectness is expected but not computed.
        """
        objectness, confidences, bboxes = [], [], []

        with torch.set_grad_enabled(self._require_grad):
            for i, c in enumerate(inpt):
                output = self._model(c)[0]
                bboxes.append(output[:, :4, :])

                objectness_chunk = (
                    output[:, 4, :]
                    if self._objectness_exists
                    else torch.zeros(output.shape[0], output.shape[2], device=output.device)
                )
                objectness.append(objectness_chunk)

                idx_start = 5 if self._objectness_exists else 4
                conf_chunk = output[:, idx_start:, :]
                confidences.append(conf_chunk)

        objectness = torch.cat(objectness, dim=0)
        confidences = torch.cat(confidences, dim=0)
        bboxes = torch.cat(bboxes, dim=0)

        objectness, confidences, bboxes = self._optional_filter_classes(
            objectness, confidences, bboxes
        )
        objectness, confidences, bboxes = self._optional_apply_nms(objectness, confidences, bboxes)

        """Here we sort all predictions in the image by the magnitude of confidences."""
        top_score, top_class = confidences.max(dim=1)
        sort = top_score.argsort(dim=1, descending=True)

        selected = []
        if self._return_confidences:
            selected.append(confidences)
        if self._return_bbox:
            selected.append(bboxes)
        if self._return_objectness:
            if objectness.numel() > 0:
                selected.append(objectness.unsqueeze(1))
            else:
                raise ValueError("Objectness expected but not computed.")
        if not selected:
            raise ValueError("No output selected.")

        data = torch.cat(selected, dim=1)
        sorted_data = torch.gather(
            data, dim=2, index=sort.unsqueeze(1).expand(-1, data.size(1), -1)
        )

        """We select which indices to return, if we only return a singular datapoint, we remove the datapoint dimension."""
        return_indices = self._return_indices or torch.arange(sorted_data.shape[-1])
        return_data = sorted_data[:, :, return_indices]

        final_return = return_data if len(return_indices) > 1 else return_data.squeeze(-1)
        return final_return

    def input_valid(self, inpt: Tensor, cond: int) -> tuple[bool, Any]:
        """
        Validate input for YOLO:

        :param inpt: Input tensor.
        :param cond: The condition to check against.
        :returns: Whether the input is valid and the prediction of the SUT.
        :raises NotImplementedError: If not implemented.
        """
        if self._return_confidences:
            pred = self.process_input(inpt)[:, :80, :]  # B x Measurements x Detections
            assert (
                pred.size(0) == 1
            ), f"Error: Expected Batch size of 1, but got {pred.size(0)}, please adapt function to work with bigger batch sizes."
            pred_am = pred.argmax(dim=1).flatten()  # B x M x D -> B x D -> D
            found = (pred_am[:5] == cond).any().item()
            return found, pred
        else:
            raise NotImplementedError(
                "Input validation not implemented for yolo without returning confidences."
            )

    def _optional_filter_classes(
        self, objectness: Tensor, confidences: Tensor, bboxes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Filter the predictions for relevant classes if applicable.

        :param objectness: The objectness to filter.
        :param confidences: The confidences to filter.
        :param bboxes: The bboxes to filter.
        :return: The filtered objectness, confidences and bboxes.
        """
        if self._filter_classes is not None:
            assert confidences.shape[0] == 1, "Error: Only implemented for Batch size = 1"
            for b in range(confidences.shape[0]):  # iterate batch
                clss = confidences[0].argmax(dim=0)
                conf = confidences[0].max(dim=0).values

                mask = torch.zeros_like(clss, dtype=torch.bool)
                for c in self._filter_classes:
                    mask |= clss == c
                mask &= conf > 0.5

                indices = mask.nonzero(as_tuple=True)[0]

                if len(indices) == 0:
                    objectness = torch.empty((1, 0), device=objectness.device)
                    confidences = torch.empty(
                        (1, confidences.size(1), 0), device=confidences.device
                    )
                    bboxes = torch.empty((1, bboxes.size(1), 0), device=bboxes.device)
                else:
                    objectness = objectness[:, indices]
                    confidences = confidences[:, :, indices]
                    bboxes = bboxes[:, :, indices]

        return objectness, confidences, bboxes

    def _optional_apply_nms(
        self, objectness: Tensor, confidences: Tensor, bboxes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Apply non-maximum suppression to predictions to avoid duplicates.

        :param objectness: The objectness to filter.
        :param confidences: The confidences to filter.
        :param bboxes: The bboxes to filter.
        :return: The filtered objectness, confidences and bboxes.
        """
        if self._nms_threshold <= 0.0:
            return objectness, confidences, bboxes

        B, C, N = confidences.shape
        assert B == 1, "Only implemented for batch size 1"

        if N == 0:
            return objectness, confidences, bboxes

        # Convert bboxes to [N,4] for NMS
        # bboxes: [B,4,N] -> [N,4]
        bboxes_N4 = bboxes[0].permute(1, 0)  # [4,N] -> [N,4]
        cx, cy, w, h = bboxes_N4[:, 0], bboxes_N4[:, 1], bboxes_N4[:, 2], bboxes_N4[:, 3]
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)  # [N,4]

        # Get max confidence per detection
        max_conf, best_class = confidences[0].max(dim=0)  # [N]

        keep_indices = []
        for c in best_class.unique():
            idxs = (best_class == c).nonzero(as_tuple=True)[0]
            kept = nms(boxes[idxs], max_conf[idxs], iou_threshold=self._nms_threshold)
            kept = torch.as_tensor(kept, device=boxes.device, dtype=torch.long)
            if kept.numel() > 0:
                keep_indices.append(idxs[kept])

        if len(keep_indices) == 0:
            keep = torch.tensor([], device=bboxes.device, dtype=torch.long)
        else:
            keep = torch.cat(keep_indices)
        if keep.dim() == 0:
            keep = keep.unsqueeze(0)

        # Use index_select to preserve detection dim
        objectness = torch.index_select(objectness, dim=1, index=keep)
        confidences = torch.index_select(confidences, dim=2, index=keep)
        bboxes = torch.index_select(bboxes, dim=2, index=keep)

        return objectness, confidences, bboxes

    @property
    def class_mapping(self) -> dict[int, str]:
        """
        Get the internal YOLO class mapping.

        :return: Mapping from class index to class name.
        """
        mapping: dict[int, str] = self._model.names
        return mapping

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing if implemented.

        :param enable: Whether to enable gradient checkpointing.
        """
        logging.warning(
            f"Gradient checkpointing not implemented for {self.__class__.__name__}, passing on."
        )
