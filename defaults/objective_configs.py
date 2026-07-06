"""A collection of metric selections for specific objectives."""

from src.objectives.auxiliary_criteria import ArchiveSparsity
from src.objectives.classifier_criteria import (
    AdversarialDistance,
    DynamicConfidenceBalance,
    NaiveConfidenceBalance,
)
from src.objectives.image_criteria import MatrixDistance, SegMapIoU, VLMBBoxIoU
from src.objectives.text_criteria import PromptObjectDistance

"""
### Adversarial Testing:
The objective is to find inputs that induce misbehavior in the SUT, while exhibiting minimal changes to the original reference.

DYNAMIC_ADVERSARIAL_TESTING: Allows for a changing target in the optimization, i.e the adversarial class can change.
TARGETED_ADVERSARIAL_TESTING: Enforces a fixed target in the optimization, i.e the adversarial class can`t change.
"""
DYNAMIC_ADVERSARIAL_TESTING = [
    AdversarialDistance(),
    MatrixDistance(),
]
TARGETED_ADVERSARIAL_TESTING = [
    AdversarialDistance(target_pair=True, exp_decay_lambda=5.0),
    MatrixDistance(),
]

"""
### Boundary Testing:
The objective is to find ambiguous cases, that is inputs for which the SUTs most likely prediction forms an equilibrium between multiple classes.

DYNAMIC_BOUNDARY_TESTING: Allows for a changing target in the optimization.
TARGETED_BOUNDARY_TESTING: Enforces a fixed target in the optimization.
ADVERSARIAL_BOUNDARY_TESTING: Enforces a fixed target in the optimization in addition to minimizing image distance to the original input.
"""
DYNAMIC_BOUNDARY_TESTING = [DynamicConfidenceBalance()]
TARGETED_BOUNDARY_TESTING = [NaiveConfidenceBalance()]
ADVERSARIAL_BOUNDARY_TESTING = [NaiveConfidenceBalance(), MatrixDistance()]


"""
### Generic Testing
Here the objective is to find inputs that induce misbehavior in the SUT, no other behavioral restrictions are applied.

MARYAM: Taken from the paper: "Benchmarking Generative AI Models for Deep Learning Test Input Generation", which misclassification severity.
DEEP_JANUS: Taken from the paper: "Model-Based Exploration of the Frontier of Behaviours for Deep Learning System Testing". Adapted to work for generic testing, rather than frontier pair discovery.
"""
MARYAM = [AdversarialDistance()]
DEEP_JANUS = [ArchiveSparsity(metric=MatrixDistance()), AdversarialDistance()]


MMM = [VLMBBoxIoU(), MatrixDistance(), PromptObjectDistance()]


"""
### Experimental objectives:
A collection of experimental objectives that are less standard.

SPATIAL_CONSISTENT_FAILURE: Aims to maximize IoU, meaning the content structure stays the same, but the class labels change.
"""
SPATIAL_CONSISTENT_FAILURE = [
    AdversarialDistance(target_pair=True, exp_decay_lambda=5.0),
    SegMapIoU(gaussian_params=(5, 1.0)),
]
# TODO: Yolo bounding box testing? Spatial failure or classification failure.
