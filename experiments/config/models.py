"""vLLM model id and bbox output format of every VLM under test."""

from typing import Any

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen": {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "coord_scale": 1000,
        "bbox_order": "xyxy",
    },
    "kimi": {
        "model": "moonshotai/Kimi-VL-A3B-Instruct",
        "coord_scale": 1,
        "bbox_order": "xyxy",
    },
    "intern": {
        "model": "OpenGVLab/InternVL3_5-8B",
        "coord_scale": 1000,
        "bbox_order": "xyxy",
    },
    "gemma": {
        "model": "google/gemma-3-4b-it",
        "coord_scale": 1000,
        "bbox_order": "yxyx",
        "image_resize": (896, 896),
    },
    "deepseek": {
        "model": "deepseek-ai/deepseek-vl2-tiny",
        "coord_scale": 999,
        "bbox_order": "xyxy",
        "prompt_mode": "deepseek_ref",
        "max_model_len": 4096,
    },
    "nemotron": {
        "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        "coord_scale": 1000,
        "bbox_order": "xyxy",
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 128,
        },
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
}
