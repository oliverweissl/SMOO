"""A collection of criteria used for comparing images (matrices)."""

from ._bbox_iou import VLMBBoxIoU
from ._cos_dissimilarity import CosDissimilarity
from ._matrix_distance import MatrixDistance
from ._ms_ssim import MSSSIM
from ._seg_map_iou import SegMapIoU
from ._ssim_d2 import SSIMD2
from ._uqi import UQI

__all__ = [
    "SSIMD2",
    "MSSSIM",
    "UQI",
    "CosDissimilarity",
    "MatrixDistance",
    "SegMapIoU",
    "VLMBBoxIoU",
]
