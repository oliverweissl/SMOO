import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Set CUDA environment variables for custom extension compilation
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["CPATH"] = f"{os.environ['CUDA_HOME']}/include:" + os.environ.get("CPATH", "")
os.environ["LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get(
    "LIBRARY_PATH", ""
)

import argparse
import logging
from argparse import Namespace

import torch
from defaults.logging_utils import setup_logging
from defaults.mimicry import ExperimentConfig, MimicryTester

from src.objectives.image_criteria import MatrixDistance
from src.objectives.classifier_criteria import TorchLossCriterion
from defaults.optimizer_configs import PYMOO_AGE_MOEA_DEFAULT_PARAMS
from torchvision.models import Wide_ResNet50_2_Weights as wrnw
from torchvision.models import wide_resnet50_2

from models.predictors.attributes_classifier import AttributeClassifier

from src.manipulator.style_gan_manipulator import StyleGANManipulator
from src.objectives import CriterionCollection
from src.optimizer import PymooOptimizer
from src.sut import ClassifierSUT, BinaryClassifierSUT

OBJ = {
    "custom": [
        MatrixDistance(),
        TorchLossCriterion(loss_fn=torch.nn.CrossEntropyLoss(reduction="none")),
    ],
    "custom_binary": [
        MatrixDistance(),
        TorchLossCriterion(loss_fn=torch.nn.BCEWithLogitsLoss(reduction="none")),
    ]
}


def main(cargs: Namespace) -> None:
    _ = setup_logging()

    logging.info("Load and Instantiate Stuff.")
    device = torch.device(f"cuda:{cargs.gpu}" if torch.cuda.is_available() else "cpu")
    """Instantiate SMOO components."""
    match cargs.objective:
        case "custom":
            sut = ClassifierSUT(
                model=wide_resnet50_2(weights=wrnw.IMAGENET1K_V2), device=device
            )

            mix_size = (1, 15)
            manipulator = StyleGANManipulator(
                "/home/weissl/Projects/SMOO/models/generators/imagenet256.pkl",
                device,
                mix_size,
                interpolate=True,
            )

        case "custom_binary":
            aclass = AttributeClassifier()
            aclass.load_state_dict(
                torch.load("models/predictors/resnet_celeb_40_single.pth", weights_only=True)
            )

            sut = BinaryClassifierSUT(
                model=aclass,
                device=device,
                batch_size=25,
                apply_sigmoid=False,
            )
            mix_size = (0, 8)
            manipulator = StyleGANManipulator(
                "/home/weissl/Projects/SMOO/models/generators/stylegan2-ffhq-1024x1024.pkl",
                device,
                mix_size,
                interpolate=True,
                batch_size=25,
            )

            #early_termination = get_early_termination(OBJ[cargs.objective][1], lambda x: x < 0.5)
        case _:
            raise NotImplementedError(
                f"Objective {cargs.objective} has no defined SUT and Manipulator in this experiment.")

    objectives = CriterionCollection(*OBJ[cargs.objective])

    aargs = PYMOO_AGE_MOEA_DEFAULT_PARAMS
    aargs["algo_params"]["pop_size"] = cargs.pop_size
    optimizer = PymooOptimizer(
        **aargs,
        solution_shape=(mix_size[1]-mix_size[0],),  # Initialize default size.
        num_objectives=len(OBJ[cargs.objective]),
    )
    budget = (
        cargs.generations
        * len(cargs.classes)
        * cargs.num_samples
        * aargs["algo_params"]["pop_size"]
    )
    logging.info(f"Simulation budget: {budget} SUT evals.")

    """Instantiate tester class and start testing."""
    config = ExperimentConfig(
        classes=cargs.classes,
        samples_per_class=cargs.num_samples,
        generations=cargs.generations,
        save_as=f"runs/mimicry_{cargs.objective}",
    )

    tester = MimicryTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=objectives,
        config=config,
        frontier_pairs=False,
    )

    logging.info("Run tester.")
    tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mimicry tester.")
    parser.add_argument("-c", "--classes", type=int, nargs="+", required=True)
    parser.add_argument("-n", "--num_samples", type=int, required=True)
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        required=True,
    )
    parser.add_argument("-p", "--pop_size", type=int, default=100)
    parser.add_argument("-o", "--objective", type=str, default="dta", choices=OBJ.keys())
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    main(args)
