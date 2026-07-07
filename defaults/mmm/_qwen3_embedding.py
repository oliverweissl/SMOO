from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import requests  # type: ignore[import-untyped]
from vllm import LLM

_SERVED_PORT = 8699
_SERVED_URL = f"http://localhost:{_SERVED_PORT}"


def _find_embedding_server(model_id: str) -> Optional[str]:
    """Check whether a matching vLLM embedding server is running.

    :param model_id: Model identifier expected to be served by the local vLLM endpoint.
    :returns: The base URL of the embedding server when a matching model is available, else ``None``.
    """
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
        self._served_url: Optional[str] = None
        self.llm: Optional[LLM] = None

        served_url = _find_embedding_server(model_id)
        if served_url is not None:
            logging.getLogger(__name__).info(
                "Detected vLLM embedding server at %s - HTTP mode.", served_url
            )
            self._served_url = f"{served_url}/v1/embeddings"
            return

        self.llm = LLM(
            model=model_id,
            seed=seed,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def run_inference(self, text: str) -> tuple[np.ndarray, int, float]:
        """Embed one text string and return the normalized vector.

        :param text: Text to embed.
        :returns: Normalized embedding vector, token count, and runtime in seconds.
        :raises RuntimeError: If no embedding backend is initialized.
        """
        if self._served_url is not None:
            payload = {"model": self._model_id, "input": [text]}
            t0 = time.time()
            response = requests.post(self._served_url, json=payload, timeout=120)
            runtime = time.time() - t0
            response.raise_for_status()
            item = response.json()["data"][0]
            embedding = np.array(item["embedding"], dtype=np.float32)
            embedding /= np.linalg.norm(embedding) + 1e-9
            return embedding, len(text.split()), runtime

        if self.llm is None:
            raise RuntimeError("Embedding model is not initialized.")
        t0 = time.time()
        output = self.llm.embed([{"prompt": text}])[0]
        runtime = time.time() - t0
        embedding = np.array(output.outputs.embedding, dtype=np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-9
        return embedding, len(output.prompt_token_ids), runtime
