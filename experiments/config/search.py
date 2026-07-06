PARETO_FILENAME = "pareto_front.json"
BEST_RESULT_FILENAME = "best_result.json"
BEST_RESULT_IMAGE_FILENAME = "best_result.png"
BASELINE_FAIL_FILENAME = "baseline_fail.json"

IMAGE_PERTURBATIONS = [
    "elastic",  # Structural deformation on clean pixels first
    "gaussian_noise",  # Noise on clean/deformed pixels
    "defocus_blur",  # Absorbs noise, degrades spatial features
    "motion_blur",  # Compounds blur, more natural on soft input
    "fog_filter",  # Global overlay, physically plausible after blur
    "snow_filter",  # Overlay, after blur (requires auxiliary_files/snow.png)
    "contrast",  # Contrast shift after spatial corruptions
    "false_color",  # Color channel manipulation
    "grayscale",  # Color reduction
    "cutout",  # Occlusion patches (bboxes passed from evaluator)
    "pixelate",  # Locks in all prior damage at reduced resolution
    "jpeg_filter",  # ALWAYS LAST, simulates real compression
]

TEXT_PERTURBATIONS = [
    "homophone",  # Dictionary lookup, needs clean tokens
    "synonym",  # Dictionary lookup, needs clean tokens
    "ata_saliency",  # Saliency-targeted typos, needs word boundaries
    "fragmentation",  # Splits words, after all word-level ops
    "character_noise",  # ALWAYS LAST, character-level, breaks everything upstream
]
