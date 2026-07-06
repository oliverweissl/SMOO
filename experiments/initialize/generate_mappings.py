"""
Generate homophone and synonym mappings from label_mapping.json via a local Ollama model.

Reads label_mapping.json (produced by build_label_mapping.py), calls Ollama in
chunks, and writes homophone_mapping.json and synonym_mapping.json.

Usage:
    python generate_mappings.py [--model MODEL] [--host HOST] [--chunk-size N]
"""

import argparse
import json
import os
from typing import Optional

import requests
from config.experiment import MAPPING_CHUNK_SIZE, OLLAMA_MODEL
from config.paths import (
    HOMOPHONE_MAPPING_FILE,
    LABEL_MAPPING_FILE,
    OLLAMA_HOST,
    SYNONYM_MAPPING_FILE,
)
from config.prompts import HOMOPHONE_MAPPING_PROMPT, SYNONYM_MAPPING_PROMPT


def call_ollama(prompt: str, model: str, host: str) -> Optional[str]:
    """Send a single chat message to a local Ollama instance and return the text response.

    :param prompt: User message content.
    :param model: Ollama model name (e.g. ``"llama3"``).
    :param host: Ollama base URL (e.g. ``"http://localhost:11434"``).
    :returns: Decoded response text, or ``None`` on failure.
    """
    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return None


def extract_json(text: str) -> dict:
    """Extract the first complete JSON object from a free-form text string.

    :param text: Raw text that may contain a JSON object.
    :returns: Parsed dict, or an empty dict if no valid JSON object is found.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def generate_mapping(
    labels: list[str],
    prompt_template: str,
    model: str,
    host: str,
    chunk_size: int,
    max_retries: int = 5,
) -> dict:
    """Generate a label-to-alternatives mapping by querying Ollama in chunks with retries.

    :param labels: List of class-label strings to map.
    :param prompt_template: Prompt string with a ``{labels}`` placeholder.
    :param model: Ollama model name.
    :param host: Ollama base URL.
    :param chunk_size: Number of labels per LLM call.
    :param max_retries: Maximum number of retry passes for labels with empty responses.
    :returns: Dict mapping each label to a list of alternatives.
    """
    result: dict = {}
    for i in range(0, len(labels), chunk_size):
        chunk = labels[i : i + chunk_size]
        print(f"  chunk {i // chunk_size + 1}: {chunk}")
        raw = call_ollama(prompt_template.format(labels=json.dumps(chunk)), model, host)
        if raw:
            result.update(extract_json(raw))
        else:
            print(f"  Warning: no response for chunk {i // chunk_size + 1}, skipping.")

    # Retry labels that got no entries or empty lists
    for attempt in range(1, max_retries + 1):
        missing = [l for l in labels if not result.get(l)]
        if not missing:
            break
        print(f"  Retry {attempt}/{max_retries}: {len(missing)} labels still empty — {missing}")
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i : i + chunk_size]
            raw = call_ollama(prompt_template.format(labels=json.dumps(chunk)), model, host)
            if raw:
                result.update({k: v for k, v in extract_json(raw).items() if v})
    else:
        still_missing = [l for l in labels if not result.get(l)]
        if still_missing:
            print(
                f"  Warning: {len(still_missing)} labels still empty after {max_retries} retries: {still_missing}"
            )

    return result


def main() -> None:
    """CLI entry point: generate homophone and synonym mapping files via Ollama."""
    parser = argparse.ArgumentParser(description="Generate homophone/synonym mappings via Ollama")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model (default: %(default)s)")
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama host (default: %(default)s)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=MAPPING_CHUNK_SIZE,
        help="Labels per LLM call (default: %(default)s)",
    )
    args = parser.parse_args()

    with open(LABEL_MAPPING_FILE) as f:
        labels = sorted(json.load(f).keys())
    print(f"Loaded {len(labels)} labels from {LABEL_MAPPING_FILE}")

    os.makedirs(os.path.dirname(HOMOPHONE_MAPPING_FILE), exist_ok=True)

    if os.path.exists(HOMOPHONE_MAPPING_FILE):
        print(f"Skipping homophones — {HOMOPHONE_MAPPING_FILE} already exists.")
    else:
        print("Generating homophones...")
        homophones = generate_mapping(
            labels, HOMOPHONE_MAPPING_PROMPT, args.model, args.host, args.chunk_size
        )
        with open(HOMOPHONE_MAPPING_FILE, "w") as f:
            json.dump(homophones, f, indent=4)
        print(f"Saved {len(homophones)} entries to {HOMOPHONE_MAPPING_FILE}")

    if os.path.exists(SYNONYM_MAPPING_FILE):
        print(f"Skipping synonyms — {SYNONYM_MAPPING_FILE} already exists.")
    else:
        print("Generating synonyms...")
        synonyms = generate_mapping(
            labels, SYNONYM_MAPPING_PROMPT, args.model, args.host, args.chunk_size
        )
        with open(SYNONYM_MAPPING_FILE, "w") as f:
            json.dump(synonyms, f, indent=4)
        print(f"Saved {len(synonyms)} entries to {SYNONYM_MAPPING_FILE}")


if __name__ == "__main__":
    main()
