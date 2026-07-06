from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.manipulator.pertubation_manipulator import PerturbCandidate, PerturbCandidateList
from src.objectives import CriterionCollection
from src.objectives.image_criteria import VLMBBoxIoU
from src.sut import VLMSUT

BEST_RESULT_FILENAME = "best_result.json"
BEST_RESULT_IMAGE_FILENAME = "best_result.png"
BASELINE_FAIL_FILENAME = "baseline_fail.json"

logger = logging.getLogger(__name__)


def resize_image_smart(image: Image.Image, max_resolution: int) -> Image.Image:
    """Resize an image proportionally when its longest side exceeds the configured limit."""
    width, height = image.size
    if max(width, height) <= max_resolution:
        return image

    scale = max_resolution / float(max(width, height))
    resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(resized, Image.Resampling.LANCZOS)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract the first JSON array contained in a VLM response or fail explicitly."""
    start = text.find("[")
    if start < 0:
        raise ValueError(f"VLM response does not contain a JSON array: {text[:400]!r}")

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                snippet = text[start : idx + 1]
                try:
                    payload = json.loads(snippet)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to decode VLM JSON array: {snippet[:400]!r}") from exc
                if not isinstance(payload, list):
                    raise TypeError(f"Expected VLM JSON payload to be a list, got {type(payload).__name__}.")
                for item in payload:
                    if not isinstance(item, dict):
                        raise TypeError(
                            f"Expected each VLM prediction to be a dict, got {type(item).__name__}."
                        )
                return payload

    raise ValueError(f"VLM response contains an unterminated JSON array: {text[:400]!r}")


def extract_target_objects(prompt: str) -> list[str]:
    """Extract the requested object labels from the MMM detection prompt."""
    match = re.search(r'Detect the object\(s\)\s+"([^"]+)"', prompt)
    if match is None:
        match = re.search(r'"([^"]+)"', prompt)
    if match is None:
        raise ValueError(f"Could not extract target objects from prompt: {prompt!r}")

    objects = [item.strip() for item in match.group(1).split(',') if item.strip()]
    if not objects:
        raise ValueError(f"Extracted no target objects from prompt: {prompt!r}")
    return objects


def build_prompt_template(prompt: str, objects: list[str]) -> str:
    """Convert a concrete detection prompt into the template expected by the SMOO manipulator."""
    object_str = ", ".join(objects)
    target = f'"{object_str}"'
    if target not in prompt:
        raise ValueError(
            f"Prompt does not contain the extracted object string {target!r}: {prompt!r}"
        )
    return prompt.replace(target, '"{objects}"', 1)


def load_sample(folder_path: str | Path, max_resolution: int) -> dict[str, Any]:
    """Load one selected MMM sample into a runtime dict."""
    folder = Path(folder_path)
    input_json = folder / 'original.json'
    input_img = folder / 'data_point.JPEG'
    if not input_json.exists() or not input_img.exists():
        raise FileNotFoundError(f'Missing original.json or data_point.JPEG in {folder}')

    with input_json.open('r', encoding='utf-8') as handle:
        base_data = json.load(handle)

    if 'prompt' not in base_data:
        raise KeyError(f"Missing 'prompt' in {input_json}")
    if 'ground_truth' not in base_data or not isinstance(base_data['ground_truth'], dict):
        raise KeyError(f"Missing or invalid 'ground_truth' in {input_json}")

    raw_img = Image.open(input_img).convert('RGB')
    clean_image = resize_image_smart(raw_img, max_resolution)
    objects = extract_target_objects(base_data['prompt'])
    return {
        'folder_path': str(folder),
        'filename': base_data.get('image', input_img.name),
        'clean_image_pil': clean_image,
        'orig_dims': raw_img.size,
        'curr_dims': clean_image.size,
        'original_prompt': base_data['prompt'],
        'prompt_template': build_prompt_template(base_data['prompt'], objects),
        'objects': objects,
        'gt_bboxes': base_data['ground_truth'],
        'baseline_iou': float(base_data.get('IoU', 0.0)),
    }


def get_output_dir(category: str, folder_id: str, output_base: str | Path) -> str:
    """Build the output directory path for one MMM sample."""
    return os.path.join(str(output_base), category, folder_id)


def save_baseline_fail(output_dir: str | Path, baseline_iou: float, sample_data: dict[str, Any]) -> None:
    """Persist the baseline-failure record for a clean-image VLM miss."""
    os.makedirs(output_dir, exist_ok=True)
    record = {
        'status': 'baseline_fail',
        'baseline_iou': float(f'{baseline_iou:.5f}'),
        'data_source': {
            'folder_path': sample_data['folder_path'],
            'folder_id': sample_data['folder_id'],
            'category': sample_data['category'],
            'filename': sample_data['filename'],
        },
        'original_prompt': sample_data['original_prompt'],
        'ground_truth_bboxes': sample_data['gt_bboxes'],
        'predicted_bboxes': sample_data['baseline_preds'],
    }
    with open(Path(output_dir) / BASELINE_FAIL_FILENAME, 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)


def active_solution_shape(mode: str, image_dim: int, text_dim: int) -> tuple[int, ...]:
    """Return the optimizer genome shape required for the selected MMM mode."""
    if image_dim <= 0:
        raise ValueError(f'Invalid image_dim: {image_dim}')
    if text_dim <= 0:
        raise ValueError(f'Invalid text_dim: {text_dim}')
    if mode == 'image':
        return (image_dim,)
    if mode == 'text':
        return (text_dim,)
    if mode == 'multi':
        return (image_dim + text_dim,)
    raise ValueError(f'Unsupported MMM mode: {mode}')


def build_population_candidates(
    genomes: np.ndarray,
    sample_data: dict[str, Any],
    mode: str,
    image_dim: int,
    text_dim: int,
) -> PerturbCandidateList:
    """Create the immutable perturbation candidate list for one optimizer population."""
    if genomes.ndim == 1:
        genomes = genomes.reshape(1, -1)
    if genomes.ndim != 2:
        raise ValueError(f'Expected genomes to be 2D, got shape {genomes.shape}.')

    candidates = []
    for genome in genomes:
        vector = np.asarray(genome, dtype=float).reshape(-1)
        if mode == 'image':
            if vector.size != image_dim:
                raise ValueError(f'Image genome size {vector.size} does not match image_dim {image_dim}.')
            image_genome = vector.tolist()
            text_genome = [0.0] * text_dim
        elif mode == 'text':
            if vector.size != text_dim:
                raise ValueError(f'Text genome size {vector.size} does not match text_dim {text_dim}.')
            image_genome = [0.0] * image_dim
            text_genome = vector.tolist()
        elif mode == 'multi':
            if vector.size != image_dim + text_dim:
                raise ValueError(
                    f'Multimodal genome size {vector.size} does not match image_dim + text_dim {image_dim + text_dim}.'
                )
            image_genome = vector[:image_dim].tolist()
            text_genome = vector[image_dim : image_dim + text_dim].tolist()
        else:
            raise ValueError(f'Unsupported MMM mode: {mode}')

        boxes = [
            [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
            for bbox in sample_data['gt_bboxes'].values()
        ]
        candidates.append(
            PerturbCandidate(
                prompt=sample_data['prompt_template'],
                objects=sample_data['objects'],
                image=os.path.join(sample_data['folder_path'], 'data_point.JPEG'),
                original_bboxes=boxes,
                text_perturbation=text_genome,
                image_pertubation=image_genome,
                image_array=np.array(sample_data['clean_image_pil'], copy=True),
            )
        )
    return PerturbCandidateList(*candidates)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure a numpy image is RGB-shaped."""
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected RGB-like image array, got shape {image.shape}.')
    return image


def image_to_tensor(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Convert an image to a float tensor in CHW layout and [0, 1] range."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    array = ensure_rgb(np.asarray(image, dtype=np.uint8))
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def _extract_bbox(pred: dict[str, Any]) -> list[float]:
    for key in ('bbox', 'bbox_2d', 'bounding_box', 'box'):
        if key in pred:
            bbox = pred[key]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f'Invalid bbox payload under key {key!r}: {bbox!r}')
            return [float(value) for value in bbox]
    raise KeyError(f'Prediction is missing bbox field: {pred!r}')


def _extract_label(pred: dict[str, Any]) -> str:
    for key in ('label', 'object', 'class', 'name', 'category'):
        if key in pred:
            value = str(pred[key]).strip()
            if not value:
                raise ValueError(f'Prediction label under key {key!r} is empty: {pred!r}')
            return value
    raise KeyError(f'Prediction is missing label field: {pred!r}')


def _normalise_label(label: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', label.lower()).strip()


def _labels_match(pred_label: str, gt_label: str, valid_prompt_labels: list[str] | None) -> bool:
    pred_norm = _normalise_label(pred_label)
    gt_norm = _normalise_label(gt_label)
    if not pred_norm or not gt_norm:
        raise ValueError(f'Cannot compare empty labels: pred={pred_label!r}, gt={gt_label!r}')
    if pred_norm == gt_norm:
        return True

    pred_tokens = set(pred_norm.split())
    gt_tokens = set(gt_norm.split())
    if pred_tokens and gt_tokens and (pred_tokens <= gt_tokens or gt_tokens <= pred_tokens):
        return True

    for label in valid_prompt_labels or []:
        label_norm = _normalise_label(label)
        label_tokens = set(label_norm.split())
        if pred_norm == label_norm:
            return True
        if pred_tokens and label_tokens and (pred_tokens <= label_tokens or label_tokens <= pred_tokens):
            return True
    return False


def _to_pixel_box(
    bbox: list[float],
    ref_w: int,
    ref_h: int,
    coord_scale: int | None,
    bbox_order: str,
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f'Expected bbox of length 4, got {bbox!r}')
    a, b, c, d = bbox
    if bbox_order == 'yxyx':
        x1, y1, x2, y2 = b, a, d, c
    elif bbox_order == 'xyxy':
        x1, y1, x2, y2 = a, b, c, d
    else:
        raise ValueError(f'Unsupported bbox order: {bbox_order}')
    scale = float(coord_scale) if coord_scale else 1.0
    return [x1 * ref_w / scale, y1 * ref_h / scale, x2 * ref_w / scale, y2 * ref_h / scale]


def prepare_bbox_pairs(
    gt_dict: dict[str, Any],
    pred_list: list[Any],
    ref_w: int,
    ref_h: int,
    valid_prompt_labels: list[str] | None,
    coord_scale: int | None,
    bbox_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align GT boxes with the best matching predicted box for each object label."""
    if not isinstance(pred_list, list):
        raise TypeError(f'Expected parsed predictions to be a list, got {type(pred_list).__name__}.')

    gt_boxes = []
    pred_boxes = []
    iou_metric = VLMBBoxIoU()

    for key, bbox in gt_dict.items():
        gt_label = key.split('_')[0]
        gt_box = np.array([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], dtype=np.float64)

        best_pred = np.zeros(4, dtype=np.float64)
        best_iou = -1.0
        for pred in pred_list:
            if not isinstance(pred, dict):
                raise TypeError(f'Expected prediction dict, got {type(pred).__name__}.')
            pred_bbox = _extract_bbox(pred)
            pred_label = _extract_label(pred)
            if not _labels_match(pred_label, gt_label, valid_prompt_labels):
                continue

            pred_box = np.array(
                _to_pixel_box(pred_bbox, ref_w, ref_h, coord_scale, bbox_order),
                dtype=np.float64,
            )
            current_iou = iou_metric.evaluate(boxes=[pred_box, gt_box])
            if current_iou > best_iou:
                best_iou = current_iou
                best_pred = pred_box

        gt_boxes.append(gt_box)
        pred_boxes.append(best_pred)

    if not gt_boxes:
        empty = np.zeros((0, 4), dtype=np.float64)
        return empty, empty
    return np.vstack(pred_boxes), np.vstack(gt_boxes)


def hash_text_embedding(labels: list[str], dim: int = 256) -> np.ndarray:
    """Build a deterministic lightweight embedding for object-label sequences."""
    vector = np.zeros(dim, dtype=np.float64)
    if not labels:
        raise ValueError('Expected at least one label for text embedding.')

    for label in labels:
        clean = _normalise_label(label).replace(' ', '_')
        if not clean:
            raise ValueError(f'Cannot build embedding for empty label: {label!r}')
        padded = f'^{clean}$'
        ngrams = [padded[i : i + 3] for i in range(max(1, len(padded) - 2))]
        for ngram in ngrams:
            digest = hashlib.sha1(ngram.encode('utf-8')).digest()
            vector[int.from_bytes(digest[:8], 'big') % dim] += 1.0

    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError(f'Failed to produce a non-zero embedding for labels: {labels!r}')
    return vector / norm


def evaluate_baseline(sut: VLMSUT, sample_data: dict[str, Any]) -> float:
    """Run baseline VLM inference on the clean sample and compute mean IoU."""
    responses = sut.process_input(([sample_data['clean_image_pil']], [sample_data['original_prompt']]))
    if len(responses) != 1:
        raise ValueError(f'Expected exactly one baseline response, got {len(responses)}.')

    parsed_preds = extract_json_array(responses[0])
    pred_boxes, gt_boxes = prepare_bbox_pairs(
        sample_data['gt_bboxes'],
        parsed_preds,
        sample_data['orig_dims'][0],
        sample_data['orig_dims'][1],
        sample_data['objects'],
        sut.coord_scale,
        sut.bbox_order,
    )
    sample_data['baseline_preds'] = parsed_preds
    if gt_boxes.shape[0] == 0:
        raise ValueError('Ground-truth bbox list is empty during baseline evaluation.')
    return float(f'{VLMBBoxIoU().evaluate(boxes=[pred_boxes, gt_boxes]):.5f}')


def _split_manipulation_results(results: tuple[list[Any], ...]) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] | None = None
    prompts: list[str] | None = None
    for result in results:
        if not result:
            raise ValueError('Manipulator returned an empty modality result list.')
        first = result[0]
        if isinstance(first, str):
            prompts = list(result)
        else:
            images = [ensure_rgb(np.asarray(item, dtype=np.uint8)) for item in result]

    if images is None or prompts is None:
        raise ValueError('MultimodalManipulator must return one image list and one prompt list.')
    if len(images) != len(prompts):
        raise ValueError(
            f'MultimodalManipulator returned mismatched images/prompts: {len(images)} vs {len(prompts)}.'
        )
    return images, prompts


def _coerce_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f'Expected scalar ndarray objective value, got shape {value.shape}.')
        return float(value.reshape(-1)[0])
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f'Expected singleton objective list, got {value!r}.')
        return float(value[0])
    return float(value)


def evaluate_population(
    objectives: CriterionCollection,
    manipulator: Any,
    sut: VLMSUT,
    candidates: PerturbCandidateList,
    sample_data: dict[str, Any],
) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Evaluate one optimizer population and return objective arrays plus per-candidate artifacts."""
    manipulated = manipulator.manipulate(candidates)
    if not isinstance(manipulated, tuple):
        raise TypeError(
            f'MultimodalManipulator.manipulate must return a tuple of modality outputs, got {type(manipulated).__name__}.'
        )
    images_np, prompts = _split_manipulation_results(manipulated)
    images_pil = [Image.fromarray(image.astype(np.uint8)) for image in images_np]

    start_time = time.time()
    responses = sut.process_input((images_pil, prompts))
    elapsed = time.time() - start_time
    if len(responses) != len(images_pil):
        raise ValueError(
            f'VLM returned {len(responses)} responses for {len(images_pil)} input candidates.'
        )

    clean_tensor = image_to_tensor(sample_data['clean_image_pil'])
    objective_values = {name: [] for name in objectives.names}
    artifacts: list[dict[str, Any]] = []

    for image_np, image_pil, prompt, response in zip(images_np, images_pil, prompts, responses):
        parsed_preds = extract_json_array(response)
        prompt_objects = extract_target_objects(prompt)
        pred_boxes, gt_boxes = prepare_bbox_pairs(
            sample_data['gt_bboxes'],
            parsed_preds,
            sample_data['orig_dims'][0],
            sample_data['orig_dims'][1],
            prompt_objects,
            sut.coord_scale,
            sut.bbox_order,
        )
        adv_tensor = image_to_tensor(image_np)
        objectives.evaluate_all(
            boxes=[pred_boxes, gt_boxes],
            images=[clean_tensor.unsqueeze(0), adv_tensor.unsqueeze(0)],
            embeddings=[hash_text_embedding(sample_data['objects']), hash_text_embedding(prompt_objects)],
        )

        per_candidate_results = {name: _coerce_scalar(value) for name, value in objectives.results.items()}
        missing_names = [name for name in objectives.names if name not in per_candidate_results]
        if missing_names:
            raise KeyError(f'Missing objective results for: {missing_names}')

        for name, value in per_candidate_results.items():
            objective_values[name].append(value)

        artifacts.append(
            {
                'image': image_pil,
                'prompt': prompt,
                'prompt_objects': prompt_objects,
                'response': response,
                'parsed_predictions': parsed_preds,
                'results': per_candidate_results,
                'runtime_seconds': elapsed / len(images_np),
            }
        )

    return (
        tuple(np.asarray(objective_values[name], dtype=np.float64) for name in objectives.names),
        {name: np.asarray(values, dtype=np.float64) for name, values in objective_values.items()},
        artifacts,
    )


def save_best_result(
    output_dir: str | Path,
    sample_data: dict[str, Any],
    best_candidate: Any,
    runtime: float,
    generations_completed: int,
    early_stop_generation: int | None,
) -> None:
    """Persist the selected best MMM testcase and its metadata."""
    os.makedirs(output_dir, exist_ok=True)

    payload = best_candidate.data[0] if isinstance(best_candidate.data, tuple) else best_candidate.data
    required_payload_keys = {'image', 'prompt', 'response', 'parsed_predictions', 'runtime_seconds'}
    missing_payload_keys = required_payload_keys.difference(payload)
    if missing_payload_keys:
        raise KeyError(f'Best-candidate payload is missing keys: {sorted(missing_payload_keys)}')

    fitness = [float(value) for value in best_candidate.fitness]
    l2_distance = float(np.linalg.norm(np.asarray(fitness, dtype=np.float64)))
    record = {
        'data_source': {
            'folder_path': sample_data['folder_path'],
            'folder_id': sample_data['folder_id'],
            'category': sample_data['category'],
            'filename': sample_data['filename'],
        },
        'runtime': runtime,
        'generations_completed': generations_completed,
        'early_stop_generation': early_stop_generation,
        'baseline_iou': float(f"{sample_data['baseline_iou']:.5f}"),
        'genome': np.asarray(best_candidate.solution).reshape(-1).tolist(),
        'objectives': {
            'iou': float(f'{fitness[0]:.5f}'),
            'img_dist': float(f'{fitness[1]:.5f}'),
            'txt_dist': float(f'{fitness[2]:.5f}'),
            'txt_sim': float(f'{1.0 - fitness[2]:.5f}'),
        },
        'l2_distance': float(f'{l2_distance:.5f}'),
        'original_prompt': sample_data['original_prompt'],
        'ground_truth_bboxes': sample_data['gt_bboxes'],
        'vlm_output': {
            'perturbed_prompt': payload['prompt'],
            'raw_response': payload['response'],
            'parsed_predictions': payload['parsed_predictions'],
            'runtime_seconds': payload['runtime_seconds'],
        },
    }

    with open(Path(output_dir) / BEST_RESULT_FILENAME, 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    payload['image'].save(Path(output_dir) / BEST_RESULT_IMAGE_FILENAME)
