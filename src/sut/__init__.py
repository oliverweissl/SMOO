"""The module for SUT models."""

from ..utils.optional_import import optional_import
from ._binary_classifier_sut import BinaryClassifierSUT
from ._classifier_sut import ClassifierSUT
from ._sut import SUT

YoloSUT = optional_import("src.sut._yolo_sut", "YoloSUT")

__all__ = [
    "SUT",
    "ClassifierSUT",
    "BinaryClassifierSUT",
    "YoloSUT",
]
