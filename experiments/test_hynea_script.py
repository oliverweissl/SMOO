import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from argparse import Namespace

import torch
import numpy as np
import random
from defaults.hynea import HyNeATester, ExperimentConfig
from defaults.logging_utils import setup_logging
from torchvision.models import wide_resnet50_2, efficientnet_v2_m, vit_l_16

from src.manipulator.diffusion_manipulator import SitHyNeAManipulator, LDMHyNeAManipulator, SDCNHyNeAManipulator
from src.objectives import CriterionCollection
from src.objectives.image_criteria import MatrixDistance
from src.objectives.classifier_criteria import TorchLossCriterion
from src.optimizer import TorchModelOptimizer

from models.predictors.attributes_classifier import AttributeClassifier

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.sut import ClassifierSUT, BinaryClassifierSUT, YoloSUT

OBJ = {
    "custom": [
        MatrixDistance(return_tensor=True),
        TorchLossCriterion(loss_fn=torch.nn.CrossEntropyLoss(reduction="none")),
    ],
    "custom_binary": [
        MatrixDistance(return_tensor=True),
        TorchLossCriterion(loss_fn=torch.nn.BCEWithLogitsLoss(reduction="none")),
    ],
    "custom_yolo": [
        MatrixDistance(return_tensor=True),
        TorchLossCriterion(loss_fn=torch.nn.CrossEntropyLoss(reduction="mean")),
    ]
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(cargs: Namespace) -> None:
    _ = setup_logging()

    logging.info("Load and Instantiate Stuff.")
    device = torch.device(f"cuda:{cargs.gpu}" if torch.cuda.is_available() else "cpu")
    """Instantiate SMOO components."""

    seed_str = ""
    if cargs.seed is not None:
        set_seed(cargs.seed)
        seed_str = f"seed_{cargs.seed}"

    match cargs.objectives:
        case "custom":
            match cargs.sut:
                case "default":
                    model = wide_resnet50_2(weights="IMAGENET1K_V2")
                case "vit":
                    model = vit_l_16(weights="IMAGENET1K_V1")
                case "eff":
                    model = efficientnet_v2_m(weights="IMAGENET1K_V1")
                case _:
                    raise NotImplementedError(f"No SUT defined for {cargs.sut}")
            sut = ClassifierSUT(
                model=model,
                device=device,
                require_grad=True,
            )
            manipulator = SitHyNeAManipulator(
                model_file="/home/weissl/Projects/SMOO/models/generators/ldm_im.pt",
                device=device,
                control_shape=(1000,),
            )
        case "custom_binary":
            aclass = AttributeClassifier()
            aclass.load_state_dict(
                torch.load("models/predictors/resnet_celeb_40_single.pth", weights_only=True)
            )

            sut = BinaryClassifierSUT(
                model=aclass,
                device=device,
                require_grad=True,
                apply_sigmoid=False,
            )

            manipulator = LDMHyNeAManipulator(
                control_shape=(40,),
                device=device,
                diffusion_steps=100,
            )
        case "custom_yolo":
            sut = YoloSUT(
                "yolov8n.pt",
                device=device,
                return_confidences=True,
                objectness_exists=False,
                require_grad=True,
            )

            image_size = 512  # -> Size of the image that SD generates (used to determine control size)
            yolo_detections = int(sum((image_size/s)**2 for s in [32,16,8]))  # YOLO has fixed amount of detections based on filters.
            manipulator = SDCNHyNeAManipulator(
                pipeline_path="runwayml/stable-diffusion-v1-5",
                controlnet_path="/home/weissl/Projects/DrivingDiff/checkpoint-1500/controlnet",
                control_shape=(80, yolo_detections),
                device=device,
            )
        case _:
            raise NotImplementedError(f"Objective {cargs.objectives} has no defined SUT and Manipulator in this experiment.")

    objectives = CriterionCollection(*OBJ[cargs.objectives])

    optimizer = TorchModelOptimizer(
        grad_optimizer=AdamW,
        grad_optimizer_params={"lr": cargs.lr},
        num_objectives=objectives.num_objectives,
        loss_reductor=lambda x: torch.cat([f.flatten() for f in x if f.numel() > 0]).sum(),
        scheduler=OneCycleLR,
        scheduler_params={"max_lr": cargs.max_lr, "total_steps":cargs.generations*cargs.pop_size},
    )

    logging.info("Load and Instantiate Stuff.")
    config = ExperimentConfig(
        classes=list(cargs.classes),
        samples_per_class=cargs.num_samples,
        generations=cargs.generations,
        pop_size=cargs.pop_size,
        save_as=f"{cargs.path}runs/hynea_{cargs.objectives}" + ("" if cargs.sut == "default" else f"_{cargs.sut}") + ("" if cargs.seed is None else f"_{seed_str}"),
        run_targeted=cargs.targeted,
        optimizer_schedule=list(),  # We dont have a schedule here
    )

    tester = HyNeATester(
        manipulator=manipulator,
        sut=sut,
        optimizer=optimizer,
        objectives=objectives,
        config=config
    )
    logging.info("Run tester.")
    tester.test()


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser(description="Run HyNeA tester.")
    parser.add_argument("-c", "--classes", type=int, nargs="+", required=True)
    parser.add_argument("-n", "--num_samples", type=int, required=True)
    parser.add_argument("-g", "--generations", type=int, required=True)
    parser.add_argument("-p", "--pop_size", type=int, default=100)
    parser.add_argument("-o", "--objectives", type=str, required=True)
    parser.add_argument("-ut", "--untargeted", action="store_false", dest="targeted")
    parser.add_argument("--sut", type=str, default="default", help="The SUT to test.")
    parser.add_argument( "--lr", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument( "--max_lr", type=float, default=1e-4, help="Max learning rate.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.set_defaults(targeted=True)

    args = parser.parse_args()
    main(args)
