from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """A dataclass to unify experimental configurations."""

    seed: int

    generations: int  # How many generations to optimize for.
    pop_size: int  # The size of a population in a generation.

    budget_max: float
    baseline_iou: float
    mode: str # What mode of manipulation

    save_as: str  # just a string for better saving
    results_dir: str = "results"
    selection_dir: str = "selection"

    solution_shape: Optional[tuple[int, ...]] = (
        None,
    )  # The shape of the final solution (not partial solutions).
