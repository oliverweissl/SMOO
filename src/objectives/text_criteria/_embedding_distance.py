from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .._criterion import Criterion


def _as_vector(value: Any) -> NDArray[np.float64]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("EmbeddingDistance received an empty embedding vector.")
    return cast(NDArray[np.float64], array)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 0.0:
        raise ValueError("Encountered zero-norm embedding vector.")
    return float(np.dot(vec_a, vec_b) / denom)


class EmbeddingDistance(Criterion):
    """Cosine-distance between two text embedding vectors."""

    _name: str = "EmbeddingDistance"

    def evaluate(self, *, embeddings: list[Any], **_: Any) -> float:
        """Calculate cosine distance for a matched pair of embeddings.

        :param embeddings: Two embedding vectors to compare.
        :param _: Unused extra criterion inputs.
        :returns: Cosine distance between the two vectors.
        :raises ValueError: If the embedding count or shapes are invalid.
        """
        if len(embeddings) != 2:
            raise ValueError(
                f"EmbeddingDistance expects exactly 2 embeddings, got {len(embeddings)}."
            )

        emb_a = _as_vector(embeddings[0])
        emb_b = _as_vector(embeddings[1])
        if emb_a.shape != emb_b.shape:
            raise ValueError(
                f"EmbeddingDistance expects matching embedding shapes, got {emb_a.shape} and {emb_b.shape}."
            )
        return float(1.0 - _cosine_similarity(emb_a, emb_b))
