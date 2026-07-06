from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor

from .._criterion import Criterion


def _as_vector(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class EmbeddingDistance(Criterion):
    """Cosine-distance between text embeddings."""

    _name: str = 'PromptObjectDistance'

    def evaluate(self, *, embeddings: list[Any], **_: Any) -> float:
        """Calculate cosine distance for a matched pair of embeddings."""
        if len(embeddings) != 2:
            raise ValueError(
                f'PromptObjectDistance expects exactly 2 embeddings, got {len(embeddings)}.'
            )

        emb_a = _as_vector(embeddings[0])
        emb_b = _as_vector(embeddings[1])
        return float(1.0 - _cosine_similarity(emb_a, emb_b))


class PromptObjectDistance(EmbeddingDistance):
    """Backward-compatible MMM text-distance criterion name."""
