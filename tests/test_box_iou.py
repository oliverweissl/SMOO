"""The analysis IoU (experiments/analysis/box_iou.py) must equal the IoU used during test generation."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments.analysis.box_iou import matched_box_iou

ROOT = Path(__file__).resolve().parents[1]


def _load_vlm_bbox_iou():
    """Load ``VLMBBoxIoU`` without importing ``src/__init__`` (which needs vLLM)."""
    saved = {name: sys.modules.get(name) for name in ("src", "src.objectives", "src.objectives.image_criteria")}
    try:
        for name in saved:
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT / name.replace(".", "/"))]
            sys.modules[name] = package
        for name, path in (
            ("src.objectives._criterion", "src/objectives/_criterion.py"),
            ("src.objectives.image_criteria._bbox_iou", "src/objectives/image_criteria/_bbox_iou.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, ROOT / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.VLMBBoxIoU
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


VLMBBoxIoU = _load_vlm_bbox_iou()


def _random_boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    xy = rng.uniform(0, 400, size=(n, 2))
    wh = rng.uniform(1, 200, size=(n, 2))
    return np.hstack([xy, xy + wh])


@pytest.mark.parametrize("seed", range(50))
def test_matches_generation_iou(seed: int) -> None:
    rng = np.random.default_rng(seed)
    gt = _random_boxes(rng, int(rng.integers(1, 5)))
    pred = _random_boxes(rng, int(rng.integers(0, 6)))
    expected = VLMBBoxIoU().evaluate(boxes=[gt, pred])
    assert matched_box_iou(pred, gt) == pytest.approx(expected)


def test_extra_boxes_are_penalised() -> None:
    gt = [[0, 0, 10, 10]]
    assert matched_box_iou([[0, 0, 10, 10]], gt) == pytest.approx(1.0)
    assert matched_box_iou([[0, 0, 10, 10], [50, 50, 60, 60]], gt) == pytest.approx(0.5)
    assert matched_box_iou([], gt) == 0.0
