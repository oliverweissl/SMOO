from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import copy
import random
from typing import Any

import numpy as np
import torch

from defaults.early_termination import get_early_termination
from defaults.mmm import ExperimentConfig, MMMTester
from defaults.objective_configs import MMM as MMM_OBJECTIVES
from defaults.optimizer_configs import PYMOO_NSGA2_DEFAULT_PARAMS
from defaults.mmm.optimizer_modules import BudgetRepair, BudgetAwareSampling
from src.manipulator.pertubation_manipulator import (
    ImagePertubationManipulator,
    MultimodalManipulator,
    TextualPerturbationManipulator,
)

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from defaults.logging_utils import setup_logging
from src.objectives import CriterionCollection
from src.optimizer import PymooOptimizer
from src.sut import VLMSUT

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen": {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "coord_scale": 1000,
        "bbox_order": "xyxy",
        "prompt_mode": "plain",
        "image_resize": None,
    },
    "kimi": {
        "model": "moonshotai/Kimi-VL-A3B-Instruct",
        "coord_scale": 1,
        "bbox_order": "xyxy",
        "prompt_mode": "plain",
        "image_resize": None,
    },
    "intern": {
        "model": "OpenGVLab/InternVL3_5-8B",
        "coord_scale": 1000,
        "bbox_order": "xyxy",
        "prompt_mode": "plain",
        "image_resize": None,
    },
    "gemma": {
        "model": "google/gemma-3-4b-it",
        "coord_scale": 896,
        "bbox_order": "yxyx",
        "prompt_mode": "plain",
        "image_resize": (896, 896),
    },
    "deepseek": {
        "model": "deepseek-ai/deepseek-vl2-tiny",
        "coord_scale": 999,
        "bbox_order": "xyxy",
        "prompt_mode": "deepseek_ref",
        "image_resize": None,
        "max_model_len": 4096,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MMM experiment in SMOO.")
    parser.add_argument("--vlm", required=True, choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--mode", required=True, choices=["multi", "image", "text"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--selection-dir", default="selection")
    parser.add_argument("--save-as", default=None)
    parser.add_argument("--seed", type=int, default=42669)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=15)
    parser.add_argument("--budget-max", type=float, default=1.0)
    parser.add_argument("--baseline-iou-min", type=float, default=0.5)
    parser.add_argument("--early-stop-iou-max", type=float, default=0.35)
    parser.add_argument("--early-stop-img-dist-max", type=float, default=0.1)
    parser.add_argument("--early-stop-txt-sim-min", type=float, default=0.70)
    parser.add_argument("--max-resolution", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--served-port", type=int, default=8700)
    return parser.parse_args()


def _infer_dims(manipulator: MultimodalManipulator) -> tuple[int, int]:
    image_dim = 0
    text_dim = 0
    for inner in getattr(manipulator, "_manipulators", []):
        if hasattr(inner, "perturbations"):
            image_dim = len(inner.perturbations)
        if hasattr(inner, "obj_pertubations") and hasattr(inner, "prompt_pertubations"):
            text_dim = len(inner.obj_pertubations) + len(inner.prompt_pertubations)
    if image_dim <= 0 or text_dim <= 0:
        raise ValueError(
            f"Failed to infer MMM manipulator dimensions: image_dim={image_dim}, text_dim={text_dim}"
        )
    return image_dim, text_dim


def _solution_shape(mode: str, image_dim: int, text_dim: int) -> tuple[int, ...]:
    if mode == "image":
        return (image_dim,)
    if mode == "text":
        return (text_dim,)
    if mode == "multi":
        return (image_dim + text_dim,)
    raise ValueError(f"Unsupported MMM mode: {mode}")


def main() -> None:
    _ = setup_logging()
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ############# Instantiate Manipulator
    manipulator = MultimodalManipulator(
        manipulator_types=[ImagePertubationManipulator, TextualPerturbationManipulator],
        manipulator_args=[{}, {}],
    )
    image_dim, text_dim = _infer_dims(manipulator)
    solution_shape = _solution_shape(args.mode, image_dim, text_dim)

    ############# Instantiate SUT
    spec = copy.deepcopy(MODEL_SPECS[args.vlm])
    max_model_len = (
        args.max_model_len if args.max_model_len is not None else spec.pop("max_model_len", None)
    )
    sut = VLMSUT(
        model=spec["model"],
        coord_scale=spec["coord_scale"],
        bbox_order=spec["bbox_order"],
        prompt_mode=spec["prompt_mode"],
        image_resize=spec["image_resize"],
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        max_model_len=max_model_len,
        seed=args.seed,
        served_ports=(args.served_port,),
    )

    ############# Instantiate Objectives
    objectives = CriterionCollection(*MMM_OBJECTIVES)

    ############# Instantiate Optimizer
    params = copy.deepcopy(PYMOO_NSGA2_DEFAULT_PARAMS)
    params["algo_params"]["pop_size"] = args.pop_size
    params["sampling"] = BudgetAwareSampling(args.budget_max, args.mode, image_dim, text_dim)
    params["repair"] = BudgetRepair(args.budget_max, args.mode, image_dim, text_dim)
    params["crossover"] = (SBX(prob=0.9, eta=15),)
    params["mutation"] = PM(eta=20)
    optimizer = PymooOptimizer(
        bounds=params["bounds"],
        algorithm=params["algorithm"],
        algo_params=params["algo_params"],
        num_objectives=objectives.num_objectives,
        solution_shape=solution_shape,
    )

    ############# Instantiate Early Termination
    early_termination = get_early_termination(
        objectives,
        lambda values: (
            (values[0] <= args.early_stop_iou_max)
            & (values[1] >= args.early_stop_img_dist_max)
            & (values[2] >= args.early_stop_txt_sim_min)
        ),
        fulfill="any",
    )

    ############# Instantiate Config
    config = ExperimentConfig(
        seed=args.seed,
        generations=args.num_generations,
        pop_size=args.pop_size,
        budget_max=args.budget_max,
        baseline_iou=args.baseline_iou_min,
        mode=args.mode,
        save_as=args.save_as or args.vlm,
        results_dir=args.results_dir,
        selection_dir=args.selection_dir,
        max_resolution=args.max_resolution,
        solution_shape=solution_shape,
    )

    tester = MMMTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=objectives,
        config=config,
        early_termination=early_termination,
    )
    tester.test()


if __name__ == "__main__":
    main()
