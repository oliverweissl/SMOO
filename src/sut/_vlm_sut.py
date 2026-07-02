import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from vllm import LLM, SamplingParams

from ._sut import SUT

_TIMEOUT = 10  # seconds for health/models check
_DEFAULT_PORTS = (8700, 8701, 8702, 8703, 8704)


class VLMSUT(SUT):
    "A VLM SUT."

    def __init__(
        self,
        model: str,
        coord_scale: int,
        bbox_order: str = "xyxy",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_new_tokens=2048,
        seed: int = 0,
    ) -> None:
        """
        Initialize a vLLM based VLM SUT.

        :param model: The model to use.
        :param coord_scale: The number of pixels to scale the bounding box with.
        :param bbox_order: The order of the bounding box coordinates.
        :param tensor_parallel_size: The number of gpus to use.
        :param gpu_memory_utilization: The amount of GPU memory to use.
        :param max_new_tokens: The maximum number of new tokens to use.
        :param seed: The random seed.
        """
        self._max_new_tokens = max_new_tokens
        self._served_url = None
        self._model_id = model

        self._coord_scale = coord_scale
        self._bbox_order = bbox_order

        self._find_served_url()
        if self._served_url is None:  # If no vLLM server is running it, spawn it.
            llm_kwargs: dict = dict(
                model=model,
                limit_mm_per_prompt={"image": 1},
                max_model_len=4096,
                seed=seed,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                enforce_eager=True,
                tensor_parallel_size=tensor_parallel_size,
            )

            self.llm = LLM(**llm_kwargs)
            self.sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    def input_valid(self, inpt: Any, cond: Any) -> tuple[bool, Any]:
        pass

    def process_input(self, inpt: tuple[list, list]) -> list[str]:
        """Run batched inference via server (sequential) or in-process vLLM (true batch).

        :param inpt: Tuple of images list and prompts list.
        :returns: List of vlm responses as strings.
        """
        images, prompts = inpt

        if self._served_url is not None:
            # Fire all requests concurrently — vLLM server batches them internally.
            futures_map = {}
            with ThreadPoolExecutor(max_workers=len(images)) as pool:
                for idx, (image, prompt) in enumerate(zip(images, prompts)):
                    futures_map[pool.submit(self._post_served, image, prompt)] = idx
                results = [None] * len(images)
                for fut in as_completed(futures_map):
                    results[futures_map[fut]] = fut.result()
            texts = [r[0] for r in results]
            return texts

        all_messages = [self._messages(img, p) for img, p in zip(images, prompts)]
        all_outputs = self.llm.chat(
            messages=all_messages,
            sampling_params=self.sampling,
        )
        texts = [o.outputs[0].text for o in all_outputs]
        return texts

    def _messages(self, image: Image, prompt: str) -> list[dict]:
        """Build a single-turn message list with image and text.

        :param image: PIL image for the user turn.
        :param prompt: Text prompt string.
        :returns: List containing one user message dict.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._pil_to_b64_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    @staticmethod
    def _pil_to_b64_url(img: Image.Image) -> str:
        """Encode a PIL image as a data-URI string for the vLLM completions API.

        :param img: PIL image to encode.
        :returns: ``data:image/jpeg;base64,...`` URL string.
        """
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def _find_served_url(self) -> None:
        """Scan common localhost ports for a vLLM server already serving ``model_id``.

        :returns: Base URL of the matching server, or ``None`` if not found.
        """
        for port in _DEFAULT_PORTS:
            url = f"http://localhost:{port}"
            try:
                resp = requests.get(f"{url}/v1/models", timeout=_TIMEOUT)
                if resp.status_code == 200:
                    served_ids = [m["id"] for m in resp.json().get("data", [])]
                    if any(self._model_id in sid or sid in self._model_id for sid in served_ids):
                        logging.info(
                            "Detected vLLM server for %s at %s — HTTP mode.", self._model_id, url
                        )
                        self._served_url = f"{url.rstrip('/')}/v1/chat/completions"
                        return
            except requests.exceptions.RequestException:
                continue

    @property
    def coord_scale(self) -> int:
        return self._coord_scale
    @property
    def bbox_order(self) -> str:
        return self._bbox_order
