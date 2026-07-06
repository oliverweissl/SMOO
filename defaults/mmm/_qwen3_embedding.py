from __future__ import annotations

import logging
import time
from typing import Optional, Union

import numpy as np
import requests
from vllm import LLM

_SERVED_PORT = 8699
_SERVED_URL = f"http://localhost:{_SERVED_PORT}"


def _find_embedding_server(model_id: str) -> Optional[str]:
    """Check whether a vLLM embedding server is running on port 8699 for ``model_id``."""
    try:
        response = requests.get(f"{_SERVED_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            served_ids = [model["id"] for model in response.json().get("data", [])]
            if any(model_id in served_id or served_id in model_id for served_id in served_ids):
                return _SERVED_URL
    except requests.exceptions.RequestException:
        pass
    return None


class Qwen3EmbeddingInstance:
    """Qwen3-Embedding-0.6B via a running vLLM server or in-process vLLM."""

    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        seed: int,
        gpu_memory_utilization: float = 0.1,
        model_id: Optional[str] = None,
    ) -> None:
        model_id = model_id or self.MODEL_ID
        self._model_id = model_id

        served_url = _find_embedding_server(model_id)
        if served_url is not None:
            logging.getLogger(__name__).info(
                "Detected vLLM embedding server at %s - HTTP mode.", served_url
            )
            self._served_url = f"{served_url}/v1/embeddings"
            self.llm = None
            return

        self._served_url = None
        self.llm = LLM(
            model=model_id,
            seed=seed,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def _embed_http(self, texts: list[str]) -> tuple[list[np.ndarray], list[int], float]:
        payload = {"model": self._model_id, "input": texts}
        t0 = time.time()
        response = requests.post(self._served_url, json=payload, timeout=120)
        runtime = time.time() - t0
        response.raise_for_status()
        data = response.json()["data"]
        vectors = []
        for item in sorted(data, key=lambda entry: entry["index"]):
            embedding = np.array(item["embedding"], dtype=np.float32)
            embedding /= np.linalg.norm(embedding) + 1e-9
            vectors.append(embedding)
        counts = [len(text.split()) for text in texts]
        return vectors, counts, runtime

    def run_inference(
            self,
            text: str,
            instruction: Optional[str] = None,
    ) -> tuple[np.ndarray, int, float]:
        vectors, token_counts, runtime = self.run_batch_inference(
            [text],
            instructions=instruction,
        )
        return vectors[0], token_counts[0], runtime

    def run_batch_inference(
            self,
            texts: list[str],
            instructions: Optional[Union[str, list[str]]] = None,
    ) -> tuple[list[np.ndarray], list[int], float]:
        full_texts = self._prepare_texts(texts, instructions)

        if self._served_url is not None:
            return self._embed_http(full_texts)

        t0 = time.time()
        outputs = self.llm.embed([{"prompt": text} for text in full_texts])
        runtime = time.time() - t0

        vectors = []
        token_counts = []

        for output in outputs:
            embedding = np.array(output.outputs.embedding, dtype=np.float32)
            embedding /= np.linalg.norm(embedding) + 1e-9

            vectors.append(embedding)
            token_counts.append(len(output.prompt_token_ids))

        return vectors, token_counts, runtime

    @staticmethod
    def _prepare_texts(
            texts: list[str],
            instructions: Optional[Union[str, list[str]]] = None,
    ) -> list[str]:
        if instructions is None:
            return texts

        if isinstance(instructions, str):
            return [f"{instructions}{text}" for text in texts]

        if isinstance(instructions, list):
            if len(instructions) != len(texts):
                raise ValueError("Instructions list length must match texts list length.")
            return [
                f"{instruction}{text}"
                for instruction, text in zip(instructions, texts)
            ]

        raise TypeError(f"Unsupported instructions type: {type(instructions).__name__}.")