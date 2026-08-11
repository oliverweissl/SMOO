"""Re-score saved MMM testcases at fixed VLM sampling temperatures."""
from __future__ import annotations
import argparse
import copy
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from defaults.logging_utils import setup_logging
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen": {"model": "Qwen/Qwen3-VL-4B-Instruct", "coord_scale": 1000, "bbox_order": "xyxy"},
    "kimi": {"model": "moonshotai/Kimi-VL-A3B-Instruct", "coord_scale": 1, "bbox_order": "xyxy"},
    "intern": {"model": "OpenGVLab/InternVL3_5-8B", "coord_scale": 1000, "bbox_order": "xyxy"},
    #"gemma": {"model": "google/gemma-3-4b-it", "coord_scale": 1000, "bbox_order": "yxyx", "image_resize": (896, 896)},
    #"deepseek": {"model": "deepseek-ai/deepseek-vl2-tiny", "coord_scale": 999, "bbox_order": "xyxy", "prompt_mode": "deepseek_ref", "max_model_len": 4096},
    "nemotron": {"model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8", "coord_scale": 1000, "bbox_order": "xyxy", "sampling_params": {"temperature": 0.0, "top_k": 1, "max_tokens": 128}, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
}


@dataclass(frozen=True)
class SavedTestcase:
    result_path: Path
    relative_dir: Path
    record: dict[str, Any]
    clean_image_path: Path
    adversarial_image_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--served-port", type=int, default=8700)
    parser.add_argument("--source-results-dir", type=Path, default=PROJECT_ROOT / "defaults/mmm/results")
    parser.add_argument("--ablation-results-dir", type=Path, default=PROJECT_ROOT / "defaults/mmm/ablation_results")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--replicate-count", type=int, default=1)
    result_mode = parser.add_mutually_exclusive_group()
    result_mode.add_argument("--overwrite", action="store_true", help="Replace existing replicate records.")
    result_mode.add_argument("--resume", action="store_true", help="Skip existing replicate records (the default).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.replicate_count < 1:
        parser.error("--replicate-count must be at least one")
    if any(value < 0 for value in args.temperatures):
        parser.error("temperatures must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least one")
    return args


def _clean_image(data_source: dict[str, Any], result_path: Path) -> Path:
    folder = Path(str(data_source["folder_path"]))
    if not folder.is_absolute():
        folder = (PROJECT_ROOT / folder).resolve()
    candidates = [folder / "data_point.JPEG"]
    if isinstance(data_source.get("filename"), str):
        candidates.append(folder / data_source["filename"])
    candidates.extend(sorted(folder.glob("data_point.*")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No clean image for {result_path} in {folder}")


def discover_testcases(source_results_dir: Path, model: str) -> list[SavedTestcase]:
    """Discover only fully saved optimizer successes (never baseline failures)."""
    model_dir = source_results_dir / model
    if not model_dir.is_dir():
        return []
    found: list[SavedTestcase] = []
    for result_path in sorted(model_dir.rglob("best_result.json")):
        image_path = result_path.with_name("best_result.png")
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            source = record["data_source"]
            assert isinstance(source, dict)
            assert isinstance(record["vlm_output"], dict)
            for key in ("original_prompt", "ground_truth_bboxes", "baseline_iou", "objectives"):
                _ = record[key]
            _ = record["vlm_output"]["perturbed_prompt"]
            clean_path = _clean_image(source, result_path)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
        except (OSError, KeyError, TypeError, ValueError, AssertionError) as exc:
            logging.warning("Skipping invalid saved testcase %s: %s", result_path, exc)
            continue
        found.append(SavedTestcase(result_path, result_path.parent.relative_to(model_dir), record, clean_path, image_path))
    return found


def _temperature_label(value: float) -> str:
    return str(value)


def output_path(base: Path, temperature: float, model: str, testcase: SavedTestcase, replicate: int) -> Path:
    return base / f"temperature_{_temperature_label(temperature)}" / model / testcase.relative_dir / f"replicate_{replicate:03d}.json"


def _score(raw: str, record: dict[str, Any], sut: "VLMSUT", original_size: tuple[int, int]) -> dict[str, Any]:
    try:
        from defaults.mmm._helpers import extract_json_array, prepare_bbox_pairs
        from src.objectives.image_criteria import VLMBBoxIoU
        predictions = extract_json_array(raw)
        pred, ground_truth = prepare_bbox_pairs(record["ground_truth_bboxes"], original_size, predictions, sut.coord_scale, sut.bbox_order)
        return {"raw_response": raw, "parsed_predictions": predictions, "iou": float(VLMBBoxIoU().evaluate(boxes=[pred, ground_truth])), "error": None}
    except Exception as exc:
        return {"raw_response": raw, "parsed_predictions": None, "iou": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def _source_iou(record: dict[str, Any], adversarial: bool) -> float | None:
    try:
        return float(record["objectives"]["iou"] if adversarial else record["baseline_iou"])
    except (KeyError, TypeError, ValueError):
        return None


def build_sut(model: str, port: int, temperature: float) -> "VLMSUT":
    from src.sut import VLMSUT
    spec = copy.deepcopy(MODEL_SPECS[model])
    return VLMSUT(model=spec["model"], coord_scale=spec.get("coord_scale"), bbox_order=spec.get("bbox_order", "xyxy"), prompt_mode=spec.get("prompt_mode", "plain"), image_resize=spec.get("image_resize"), max_model_len=spec.get("max_model_len"), served_ports=(port,), sampling_params=spec.get("sampling_params"), extra_body=spec.get("extra_body"), temperature=temperature)


def evaluate_testcase(sut: "VLMSUT", testcase: SavedTestcase, temperature: float, replicate: int) -> dict[str, Any]:
    record = testcase.record
    with Image.open(testcase.clean_image_path) as handle:
        clean = handle.convert("RGB")
        original_size = clean.size
    source_folder = Path(str(record["data_source"]["folder_path"]))
    if not source_folder.is_absolute():
        source_folder = (PROJECT_ROOT / source_folder).resolve()
    try:
        source_metadata = json.loads((source_folder / "original.json").read_text(encoding="utf-8"))
        dims = source_metadata.get("original_dims")
        if isinstance(dims, list) and len(dims) == 2:
            original_size = (int(dims[0]), int(dims[1]))
    except (OSError, ValueError, TypeError, KeyError):
        pass
    with Image.open(testcase.adversarial_image_path) as handle:
        adversarial = handle.convert("RGB")
    if clean.size != adversarial.size:
        clean = clean.resize(adversarial.size, Image.Resampling.LANCZOS)
    clean_raw, adversarial_raw = sut.process_input(([clean, adversarial], [record["original_prompt"], record["vlm_output"]["perturbed_prompt"]]))
    clean_score = _score(clean_raw, record, sut, original_size)
    adversarial_score = _score(adversarial_raw, record, sut, original_size)
    old_clean, old_adversarial = _source_iou(record, False), _source_iou(record, True)
    return {"status": "complete", "source": {"best_result_json": str(testcase.result_path), "best_result_png": str(testcase.adversarial_image_path), "clean_image": str(testcase.clean_image_path), "data_source": record["data_source"], "source_metrics": {"clean_iou": old_clean, "adversarial_iou": old_adversarial}}, "temperature": temperature, "replicate_index": replicate, "original_prompt": record["original_prompt"], "perturbed_prompt": record["vlm_output"]["perturbed_prompt"], "ground_truth_bboxes": record["ground_truth_bboxes"], "clean": clean_score, "adversarial": adversarial_score, "deltas_from_source": {"clean_iou": None if old_clean is None else clean_score["iou"] - old_clean, "adversarial_iou": None if old_adversarial is None else adversarial_score["iou"] - old_adversarial}}


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    testcases = discover_testcases(args.source_results_dir.resolve(), args.model)
    if args.limit:
        testcases = testcases[:args.limit]
    logging.info("Discovered %d saved %s testcases", len(testcases), args.model)
    if args.dry_run:
        return 0
    for temperature in args.temperatures:
        sut = build_sut(args.model, args.served_port, temperature)
        for testcase in testcases:
            for replicate in range(1, args.replicate_count + 1):
                destination = output_path(args.ablation_results_dir.resolve(), temperature, args.model, testcase, replicate)
                if destination.exists() and not args.overwrite:
                    logging.info("Skipping existing %s", destination)
                    continue
                try:
                    payload = evaluate_testcase(sut, testcase, temperature, replicate)
                except Exception as exc:
                    logging.exception("Ablation failed for %s", testcase.result_path)
                    payload = {"status": "failure", "source": {"best_result_json": str(testcase.result_path)}, "temperature": temperature, "replicate_index": replicate, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                _write(destination, payload)
    return 0


def main(argv: list[str] | None = None) -> None:
    _ = setup_logging()
    raise SystemExit(run(parse_args(argv)))

if __name__ == "__main__":
    main()
