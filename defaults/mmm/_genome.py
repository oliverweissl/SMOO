"""Map optimizer genomes to image/text perturbation candidates."""

from __future__ import annotations

import numpy as np

from src.manipulator.pertubation_manipulator import (
    MMMSample,
    PerturbCandidate,
    PerturbCandidateList,
)


def active_solution_shape(mode: str, image_dim: int, text_dim: int) -> tuple[int, ...]:
    """Return the optimizer genome shape required for the selected MMM mode.

    :param mode: MMM execution mode.
    :param image_dim: Number of image perturbation parameters.
    :param text_dim: Number of text perturbation parameters.
    :returns: Genome shape expected by the optimizer.
    :raises ValueError: If the mode is unsupported or one of the dimensions is invalid.
    """
    if image_dim <= 0:
        raise ValueError(f"Invalid image_dim: {image_dim}")
    if text_dim <= 0:
        raise ValueError(f"Invalid text_dim: {text_dim}")
    if mode == "image":
        return (image_dim,)
    if mode == "text":
        return (text_dim,)
    if mode == "multi":
        return (image_dim + text_dim,)
    raise ValueError(f"Unsupported MMM mode: {mode}")


def build_population_candidates(
    genomes: np.ndarray,
    sample: MMMSample,
    prompt: str,
    mode: str,
    image_dim: int,
    text_dim: int,
) -> PerturbCandidateList:
    """Create the immutable perturbation candidate list for one optimizer population.

    :param genomes: Population genome matrix.
    :param sample: MMM sample shared by the population.
    :param prompt: Prompt template for the detector.
    :param mode: MMM execution mode.
    :param image_dim: Number of image perturbation parameters.
    :param text_dim: Number of text perturbation parameters.
    :returns: Candidate list matching the optimizer population.
    :raises ValueError: If the genome shape is invalid for the selected mode.
    """
    if genomes.ndim == 1:
        genomes = genomes.reshape(1, -1)
    if genomes.ndim != 2:
        raise ValueError(f"Expected genomes to be 2D, got shape {genomes.shape}.")

    candidates = []
    for genome in genomes:
        vector = np.asarray(genome, dtype=float).reshape(-1)
        if mode == "image":
            if vector.size != image_dim:
                raise ValueError(
                    f"Image genome size {vector.size} does not match image_dim {image_dim}."
                )
            image_genome = vector.tolist()
            text_genome = [0.0] * text_dim
        elif mode == "text":
            if vector.size != text_dim:
                raise ValueError(
                    f"Text genome size {vector.size} does not match text_dim {text_dim}."
                )
            image_genome = [0.0] * image_dim
            text_genome = vector.tolist()
        elif mode == "multi":
            if vector.size != image_dim + text_dim:
                raise ValueError(
                    f"Multimodal genome size {vector.size} does not match image_dim + text_dim {image_dim + text_dim}."
                )
            image_genome = vector[:image_dim].tolist()
            text_genome = vector[image_dim : image_dim + text_dim].tolist()
        else:
            raise ValueError(f"Unsupported MMM mode: {mode}")

        candidates.append(
            PerturbCandidate(
                sample=sample,
                prompt_template=prompt,
                text_perturbation=text_genome,
                image_pertubation=image_genome,
            )
        )
    return PerturbCandidateList(*candidates)
