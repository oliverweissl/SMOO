from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """A dataclass to unify experimental configurations."""

    seed: int
    generations: int
    pop_size: int
    budget_max: float
    baseline_iou: float
    mode: str
    save_as: str

    results_dir: str = 'results'
    selection_dir: str = 'selection'
    early_stop_iou_max: float = 0.35
    early_stop_img_dist_max: float = 0.1
    early_stop_txt_sim_min: float = 0.70
    max_resolution: int = 1024
    min_perturbation_scale: float = 0.01
    solution_shape: Optional[tuple[int, ...]] = None
