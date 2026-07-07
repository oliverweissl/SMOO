from __future__ import annotations

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Literal, Optional

import numpy as np
import requests  # type: ignore[import-untyped]
from PIL import Image
from vllm import LLM, SamplingParams

from ._sut import SUT

_TIMEOUT = 10
_DEFAULT_PORTS = (8700, 8701, 8702, 8703, 8704)

TPromptMode = Literal["plain", "deepseek_ref"]
TBBoxOrder = Literal["xyxy", "yxyx"]

logger = logging.getLogger(__name__)


class VLMSUT(SUT):
    """A general vLLM-backed VLM SUT."""

    def __init__(
        self,
        model: str,
        coord_scale: Optional[int],
        bbox_order: TBBoxOrder = "xyxy",
        prompt_mode: TPromptMode = "plain",
        image_resize: Optional[tuple[int, int]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_new_tokens: int = 2048,
        max_model_len: Optional[int] = None,
        seed: int = 0,
        served_ports: tuple[int, ...] = _DEFAULT_PORTS,
    ) -> None:
        """Initialize a general vLLM-based VLM SUT.

        :param model: Hugging Face model identifier served by vLLM.
        :param coord_scale: Bounding-box coordinate scale used by the model output.
        :param bbox_order: Bounding-box coordinate order.
        :param prompt_mode: Prompt transformation mode before inference.
        :param image_resize: Optional ``(width, height)`` resize applied before encoding.
        :param tensor_parallel_size: Tensor parallelism for in-process vLLM.
        :param gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        :param max_new_tokens: Maximum generated tokens per sample.
        :param max_model_len: Optional override for vLLM model context length.
        :param seed: Random seed forwarded to vLLM.
        :param served_ports: Ports to scan for a compatible local vLLM server.
        """
        self._model_id = model
        self._coord_scale = coord_scale
        self._bbox_order = bbox_order
        self._prompt_mode = prompt_mode
        self._image_resize = image_resize
        self._max_new_tokens = max_new_tokens
        self._max_model_len = max_model_len
        self._served_ports = served_ports
        self._served_url: Optional[str] = None
        self.llm: Optional[LLM] = None
        self.sampling: Optional[SamplingParams] = None

        self._find_served_url()
        if self._served_url is None:
            llm_kwargs: dict[str, Any] = dict(
                model=model,
                limit_mm_per_prompt={"image": 1},
                max_model_len=max_model_len or 4096,
                seed=seed,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                enforce_eager=True,
                tensor_parallel_size=tensor_parallel_size,
            )
            self.llm = LLM(**llm_kwargs)
            self.sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    def input_valid(
        self, inpt: Any, cond: Any = None
    ) -> tuple[bool, tuple[list[Image.Image], list[str]]]:
        """Validate and normalize multimodal VLM input.

        :param inpt: Candidate input shaped as ``(images, prompts)``.
        :param cond: Unused validation condition placeholder.
        :returns: ``True`` and normalized ``(images, prompts)``.
        :raises ValueError: If the input structure or contained types are invalid.
        """
        if not isinstance(inpt, tuple) or len(inpt) != 2:
            raise ValueError("VLMSUT expects input shaped as (images, prompts).")

        images, prompts = inpt
        if not isinstance(images, list) or not isinstance(prompts, list):
            raise ValueError("VLMSUT expects both images and prompts to be lists.")
        if len(images) != len(prompts):
            raise ValueError(
                f"VLMSUT expects equally-sized image and prompt batches, got {len(images)} and {len(prompts)}."
            )

        normalized_images = [self._coerce_image(image) for image in images]
        normalized_prompts = []
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise ValueError(f"VLMSUT expects prompt strings, got {type(prompt).__name__}.")
            normalized_prompts.append(prompt)
        return True, (normalized_images, normalized_prompts)

    def process_input(self, inpt: tuple[list[Any], list[str]]) -> list[str]:
        """Run batched inference via an existing vLLM server or in-process vLLM.

        :param inpt: Batch input shaped as ``(images, prompts)``.
        :returns: Generated texts for the batch.
        """
        _, normalized = self.input_valid(inpt)
        images, prompts = normalized
        return self.run_batch_inference(images, prompts)[0]

    def run_inference(self, image: Any, prompt: str) -> tuple[str, int, int, float]:
        """Run single-sample inference.

        :param image: Input image.
        :param prompt: Input prompt.
        :returns: Output text, completion token count twice, and runtime in seconds.
        :raises RuntimeError: If local vLLM inference returns an invalid output structure.
        """
        _, normalized = self.input_valid(([image], [prompt]))
        images, prompts = normalized

        if self._served_url is not None:
            text, count, runtime = self._post_served(images[0], prompts[0])
            return text, count, count, runtime

        assert self.llm is not None and self.sampling is not None
        t0 = time.time()
        outputs = self.llm.chat(
            messages=self._messages(images[0], prompts[0]), sampling_params=self.sampling
        )
        runtime = time.time() - t0
        if len(outputs) != 1:
            raise RuntimeError(
                f"VLMSUT local single inference returned {len(outputs)} outputs instead of 1."
            )
        if not outputs[0].outputs:
            raise RuntimeError("VLMSUT local single inference returned no completions.")
        text = outputs[0].outputs[0].text
        count = len(outputs[0].outputs[0].token_ids)
        return text, count, count, runtime

    def run_batch_inference(
        self, images: list[Any], prompts: list[str]
    ) -> tuple[list[str], list[int], list[int], float]:
        """Run batch inference through vLLM.

        :param images: Batch of input images.
        :param prompts: Batch of input prompts.
        :returns: Output texts, completion token counts twice, and runtime in seconds.
        :raises ValueError: If the batch is empty or input validation fails.
        :raises RuntimeError: If vLLM returns an invalid output structure.
        """
        _, normalized = self.input_valid((images, prompts))
        norm_images, norm_prompts = normalized

        if self._served_url is not None:
            if not norm_images:
                raise ValueError("VLMSUT received an empty batch for HTTP inference.")
            futures_map = {}
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=len(norm_images)) as pool:
                for idx, (image, prompt) in enumerate(zip(norm_images, norm_prompts)):
                    futures_map[pool.submit(self._post_served, image, prompt)] = idx
                results: list[Optional[tuple[str, int, float]]] = [None] * len(norm_images)
                for fut in as_completed(futures_map):
                    results[futures_map[fut]] = fut.result()
            total_runtime = time.time() - t0
            if any(result is None for result in results):
                raise RuntimeError("VLMSUT HTTP batch finished with missing result slots.")
            texts = [result[0] for result in results if result is not None]
            counts = [result[1] for result in results if result is not None]
            if len(texts) != len(norm_images):
                raise RuntimeError(
                    f"VLMSUT HTTP batch returned {len(texts)} results for {len(norm_images)} inputs."
                )
            return texts, counts, counts, total_runtime

        if not norm_images:
            raise ValueError("VLMSUT received an empty batch for local inference.")

        assert self.llm is not None and self.sampling is not None
        t0 = time.time()
        all_messages = [
            self._messages(image, prompt) for image, prompt in zip(norm_images, norm_prompts)
        ]
        all_outputs = self.llm.chat(messages=all_messages, sampling_params=self.sampling)
        runtime = time.time() - t0
        if len(all_outputs) != len(norm_images):
            raise RuntimeError(
                f"VLMSUT local batch returned {len(all_outputs)} outputs for {len(norm_images)} inputs."
            )
        texts = []
        counts = []
        for output in all_outputs:
            if not output.outputs:
                raise RuntimeError("VLMSUT local batch output is missing completions.")
            texts.append(output.outputs[0].text)
            counts.append(len(output.outputs[0].token_ids))
        return texts, counts, counts, runtime

    def _coerce_image(self, image: Any) -> Image.Image:
        """Normalize supported image inputs to RGB PIL images.

        :param image: Input image as PIL image or numpy array.
        :returns: RGB PIL image.
        :raises ValueError: If the input image type or shape is unsupported.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            elif array.ndim == 3 and array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Unsupported image array shape: {array.shape}.")
            return Image.fromarray(array.astype(np.uint8), mode="RGB")
        raise ValueError(f"Unsupported image input type: {type(image).__name__}.")

    def _transform_prompt(self, prompt: str) -> str:
        """Apply model-specific prompt formatting.

        :param prompt: Input prompt.
        :returns: Transformed prompt string.
        :raises ValueError: If the configured prompt mode is unsupported.
        """
        if self._prompt_mode == "plain":
            return prompt
        if self._prompt_mode == "deepseek_ref":
            return f" <|ref|>{prompt}<|/ref|>."
        raise ValueError(f"Unsupported prompt mode: {self._prompt_mode}.")

    def _transform_image(self, image: Image.Image) -> Image.Image:
        """Apply optional model-specific image resizing.

        :param image: Input image.
        :returns: Possibly resized image.
        """
        if self._image_resize is None:
            return image
        return image.resize(self._image_resize)

    def _messages(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
        """Build a single-turn multimodal chat payload.

        :param image: Input image.
        :param prompt: Input prompt.
        :returns: vLLM-compatible multimodal chat message payload.
        """
        image = self._transform_image(image)
        prompt = self._transform_prompt(prompt)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._pil_to_b64_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _post_served(self, image: Image.Image, prompt: str) -> tuple[str, int, float]:
        """Send one request to a running vLLM server.

        :param image: Input image.
        :param prompt: Input prompt.
        :returns: Output text, completion token count, and runtime in seconds.
        :raises KeyError: If the HTTP response is missing required fields.
        :raises ValueError: If the resolved response text is empty or malformed.
        """
        assert self._served_url is not None
        payload = {
            "model": self._model_id,
            "messages": self._messages(image, prompt),
            "max_tokens": self._max_new_tokens,
            "temperature": 0,
        }
        t0 = time.time()
        response = requests.post(self._served_url, json=payload, timeout=(10, 600))
        runtime = time.time() - t0
        response.raise_for_status()
        data = response.json()
        if "choices" not in data or not isinstance(data["choices"], list) or not data["choices"]:
            raise KeyError(f"VLMSUT HTTP response is missing choices: {data!r}")
        if "message" not in data["choices"][0] or "content" not in data["choices"][0]["message"]:
            raise KeyError(f"VLMSUT HTTP response choice is missing message content: {data!r}")
        if "usage" not in data or "completion_tokens" not in data["usage"]:
            raise KeyError(f"VLMSUT HTTP response is missing usage.completion_tokens: {data!r}")
        message = data["choices"][0]["message"]["content"]
        text = self._extract_text_content(message)
        if not text:
            raise ValueError(f"VLMSUT HTTP response content resolved to an empty string: {data!r}")
        count = int(data["usage"]["completion_tokens"])
        return text, count, runtime

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Normalize OpenAI-style message content to a plain string.

        :param content: Message content payload from the served API.
        :returns: Plain text content.
        :raises ValueError: If the content structure is unsupported or resolves to an empty string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") != "text" or "text" not in item:
                        raise ValueError(
                            f"Unsupported structured content item in VLM response: {item!r}"
                        )
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    raise ValueError(
                        f"Unsupported content item type in VLM response: {type(item).__name__}."
                    )
            text = "".join(parts)
            if not text:
                raise ValueError("Structured VLM response content resolved to an empty string.")
            return text
        raise ValueError(f"Unsupported VLM response content type: {type(content).__name__}.")

    @staticmethod
    def _pil_to_b64_url(img: Image.Image) -> str:
        """Encode a PIL image as a data URI for the vLLM chat API.

        :param img: Input image.
        :returns: Base64 data URI.
        """
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def _find_served_url(self) -> None:
        """Scan configured localhost ports for a matching running vLLM server."""
        for port in self._served_ports:
            url = f"http://localhost:{port}"
            try:
                response = requests.get(f"{url}/v1/models", timeout=_TIMEOUT)
                if response.status_code != 200:
                    continue
                served_ids = [model["id"] for model in response.json().get("data", [])]
                if any(
                    self._model_id in served_id or served_id in self._model_id
                    for served_id in served_ids
                ):
                    logger.info(
                        "Detected vLLM server for %s at %s - HTTP mode.", self._model_id, url
                    )
                    self._served_url = f'{url.rstrip("/")}/v1/chat/completions'
                    return
            except requests.exceptions.RequestException:
                continue

    @property
    def coord_scale(self) -> Optional[int]:
        return self._coord_scale

    @property
    def bbox_order(self) -> TBBoxOrder:
        return self._bbox_order
