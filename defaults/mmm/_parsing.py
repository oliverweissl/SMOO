"""Parse the VLM text answer and the detection prompt."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Decode the VLM response as prediction records.

    :param text: Raw VLM response text.
    :returns: The decoded prediction list.
    :raises ValueError: If JSON decoding fails.
    :raises TypeError: If the decoded payload cannot be normalized into a list of dict objects.
    """
    raw_text = text.strip()
    fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        raw_text = fence_match.group(1).strip()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        try:
            payload = json.loads(f"[{raw_text}]")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode VLM JSON array: {text[:400]!r}") from exc

    if isinstance(payload, dict):
        if any(key in payload for key in ("bbox", "bbox_2d", "bounding_box", "box")):
            payload = [payload]
        else:
            for key in ("predictions", "objects", "detections", "results", "boxes", "output"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    payload = candidate
                    break
            else:
                raise TypeError(
                    "Expected VLM JSON payload to be a list or prediction container, "
                    f"got dict with keys {sorted(payload.keys())!r}."
                )

    if not isinstance(payload, list):
        raise TypeError(f"Expected VLM JSON payload to be a list, got {type(payload).__name__}.")
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected each VLM prediction to be a dict, got {type(item).__name__}."
            )
    return payload


def extract_target_objects(prompt: str) -> list[str]:
    """Extract the requested object labels from the MMM detection prompt.

    :param prompt: Detection prompt text.
    :returns: Parsed object labels in prompt order.
    :raises ValueError: If the prompt does not contain any extractable target objects.
    """
    match = re.search(r'Detect the object\(s\)\s+"([^"]+)"', prompt)
    if match is None:
        match = re.search(r'"([^"]+)"', prompt)
    if match is None:
        raise ValueError(f"Could not extract target objects from prompt: {prompt!r}")

    objects = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if not objects:
        raise ValueError(f"Extracted no target objects from prompt: {prompt!r}")
    return objects
