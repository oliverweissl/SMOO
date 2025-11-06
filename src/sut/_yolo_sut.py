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
            pred = self.process_input(inpt)[:80]
            found = (pred.argmax(dim=1) == cond).any().item()
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
            batch_obj, batch_conf, batch_bbox = [], [], []

            for b in range(confidences.shape[0]):  # iterate batch
                clss = confidences[b].argmax(dim=0)
                conf = confidences[b].max(dim=0).values

                mask = torch.zeros_like(clss, dtype=torch.bool)
                for c in self._filter_classes:
                    mask |= clss == c
                mask &= conf > 0.5

                indices = mask.nonzero(as_tuple=True)[0]
                logging.info(
                    f"Batch {b}: Filtered from {confidences.shape[2]} to {len(indices)} detections"
                )

                if len(indices) == 0:
                    batch_obj.append(torch.empty(0, device=objectness.device))
                    batch_conf.append(
                        torch.empty(confidences.size(1), 0, device=confidences.device)
                    )
                    batch_bbox.append(torch.empty(bboxes.size(1), 0, device=bboxes.device))
                    logging.warning(
                        f"Filtering for classes {self._filter_classes} reduced detections to 0."
                    )
                    continue

                batch_obj.append(objectness[b, indices])
                batch_conf.append(confidences[b, :, indices])
                batch_bbox.append(bboxes[b, :, indices])
            return torch.stack(batch_obj), torch.stack(batch_conf), torch.stack(batch_bbox)

        else:
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
        if self._nms_threshold > 0.0:
            batch_keep_obj, batch_keep_conf, batch_keep_bbox = [], [], []

            for b in range(confidences.shape[0]):  # batch elements
                boxes, scores, classes = [], [], []

                if confidences.shape[2] == 0:
                    logging.info(f"Element {b}: NMS skipped — no detections")
                    batch_keep_obj.append(objectness[b])
                    batch_keep_conf.append(confidences[b])
                    batch_keep_bbox.append(bboxes[b])
                    continue

                for i in range(confidences.shape[2]):  # detections
                    class_confs = confidences[b, :, i]
                    max_conf, best_class = class_confs.max(dim=0)

                    cx, cy, w, h = bboxes[b, :, i].cpu()
                    x1, y1 = cx - w / 2, cy - h / 2
                    x2, y2 = cx + w / 2, cy + h / 2

                    boxes.append([x1, y1, x2, y2])
                    scores.append(max_conf.item())
                    classes.append(best_class.item())

                boxes = torch.tensor(boxes)
                scores = torch.tensor(scores)
                classes = torch.tensor(classes)

                keep_indices = []
                for c in classes.unique():
                    idxs = (classes == c).nonzero(as_tuple=True)[0]
                    keep = nms(boxes[idxs], scores[idxs], iou_threshold=self._nms_threshold)
                    keep_indices.append(idxs[keep])

                keep = torch.cat(keep_indices)

                logging.info(
                    f"Element {b}: NMS reduced from {confidences.shape[2]} to {len(keep)} detections"
                )
                batch_keep_obj.append(objectness[b, keep])
                batch_keep_conf.append(confidences[b, :, keep])
                batch_keep_bbox.append(bboxes[b, :, keep])

            return (
                torch.stack(batch_keep_obj),
                torch.stack(batch_keep_conf),
                torch.stack(batch_keep_bbox),
            )
        else:
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
