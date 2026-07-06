from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .._manipulator import Manipulator
from ._kernels import create_disk_kernel, create_motion_blur_kernel
from ._perturb_candidate import PerturbCandidateList

BASE = Path(__file__).parent / "files"


class ImagePertubationManipulator(Manipulator):
    """A Manipulator that performs image perturbations."""

    def __init__(self, snow_file: str | Path = Path(BASE / "snow.png")) -> None:
        """
        Initialize the Manipulator.

        :param snow_file: Path to the snow image.
        """
        self._frost_path = snow_file
        self.perturbations = [
            self.jpeg_filter,
            self.pixelate,
            self.defocus_blur,
            self.motion_blur,
            self.gaussian_noise,
            self.fog_filter,
            self.snow_filter,
            self.contrast,
            self.elastic,
            self.cutout_filter_with_bbox,
            self.false_color_filter,
            self.grayscale_filter,
        ]

    def manipulate(self, candidates: PerturbCandidateList, **kwargs: Any) -> list[NDArray]:
        """
        Manipulate the candidates based on perturbations.

        :param candidates: The candidates.
        """
        for candidate in candidates:
            assert len(candidate.image_pertubation) == len(
                self.perturbations
            ), f"Error: found: {len(self.perturbations)} image perturbations, but genome is size {len(candidate.image_pertubation)}"
            kwargs = {"bboxes": candidate.original_bboxes}
            for scale, pert in zip(candidate.image_pertubation, self.perturbations):
                candidate.image_array = pert(image=candidate.image_array, scale=scale, **kwargs)

        return [c.image_array for c in candidates]

    def _interpolate(self, factor: float, values: Sequence) -> float:
        """Linear interpolation of factor ∈ [0,1] over 5 target values (discrete scales 0–4).

        :param factor: Severity in [0.0, 1.0].
        :param values: Sequence of 5 target values corresponding to scales 0–4.
        :returns: Interpolated value.
        """
        factor = max(0.0, min(1.0, factor))
        scale_anchors = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i in range(len(scale_anchors) - 1):
            if factor <= scale_anchors[i + 1]:
                slope = (values[i + 1] - values[i]) / (scale_anchors[i + 1] - scale_anchors[i])
                return values[i] + slope * (factor - scale_anchors[i])
        return values[-1]

    # -------------- Pertubations
    def jpeg_filter(
        self,
        scale: float,
        image: np.ndarray,
        quality_levels: tuple[int, ...] = (30, 18, 15, 10, 5),
        **_: Any,
    ) -> np.ndarray:
        """Apply JPEG compression artefacts at the given severity.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 BGR image.
        :param quality_levels: Five JPEG quality anchors mapped to scales 0–1.
        :returns: Re-decoded uint8 image with compression artefacts.
        """
        quality = int(self._interpolate(scale, quality_levels))
        _, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(
            np.frombuffer(BytesIO(encoded.tobytes()).read(), np.uint8), cv2.IMREAD_COLOR
        )

    def pixelate(
        self,
        scale: float,
        image: np.ndarray,
        downsample_levels: tuple[float, ...] = (0.85, 0.55, 0.35, 0.2, 0.1),
        **_: Any,
    ) -> np.ndarray:
        """Pixelate an image by downsampling then upsampling with nearest-neighbour interpolation.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param downsample_levels: Five downscale fraction anchors mapped to scales 0–1.
        :returns: Pixelated uint8 image of the original size.
        """
        downsample = self._interpolate(scale, downsample_levels)
        h, w = image.shape[:2]
        image = np.array(image, dtype=np.uint8)
        small_w, small_h = max(1, int(w * downsample)), max(1, int(h * downsample))
        return cv2.resize(
            cv2.resize(image, (small_w, small_h), cv2.INTER_AREA), (w, h), cv2.INTER_NEAREST
        )

    def defocus_blur(
        self,
        scale: float,
        image: np.ndarray,
        radius_levels: tuple[int, ...] = (2, 5, 6, 9, 12),
        **_: Any,
    ) -> np.ndarray:
        """Apply defocus blur using a disk-shaped convolution kernel.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param radius_levels: Five disk-radius anchors mapped to scales 0–1.
        :returns: Blurred uint8 image.
        """
        image = np.array(image, dtype=np.uint8)
        radius = max(1, int(self._interpolate(scale, radius_levels)))
        return cv2.filter2D(image, -1, create_disk_kernel(radius))

    def motion_blur(
        self,
        scale: float,
        image: np.ndarray,
        size_levels: tuple[int, ...] = (2, 4, 6, 10, 15),
        angle_levels: tuple[int, ...] = (5, 12, 20, 30, 45),
        **_: Any,
    ) -> np.ndarray:
        """Apply directional motion blur using a linear convolution kernel.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param size_levels: Five kernel-size anchors mapped to scales 0–1.
        :param angle_levels: Five blur-angle (degrees) anchors mapped to scales 0–1.
        :returns: Motion-blurred uint8 image.
        """
        image = np.array(image, dtype=np.uint8)
        size = max(1, int(self._interpolate(scale, size_levels)))
        angle = self._interpolate(scale, angle_levels)
        return cv2.filter2D(image, -1, create_motion_blur_kernel(size, angle))

    def gaussian_noise(
        self,
        scale: float,
        image: np.ndarray,
        std_levels: tuple[float, ...] = (0.03, 0.06, 0.12, 0.18, 0.22),
        **_: Any,
    ) -> np.ndarray:
        """Add zero-mean Gaussian noise to the image.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param std_levels: Five noise standard-deviation anchors (normalised 0–1) mapped to scales 0–1.
        :returns: Noisy uint8 image clipped to [0, 255].
        """
        image = np.array(image, dtype=np.uint8)
        std_dev = self._interpolate(scale, std_levels)
        x = image.astype(np.float32) / 255.0
        noisy = np.clip(x + np.random.normal(size=x.shape, scale=std_dev), 0, 1)
        return (noisy * 255).astype(np.uint8)

    def fog_filter(
        self,
        scale: float,
        image: np.ndarray,
        intensity_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.45, 0.65),
        noise_levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.45),
        **_: Any,
    ) -> np.ndarray:
        """Blend a noisy white overlay over the image to simulate fog.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param intensity_levels: Five blend-weight anchors mapped to scales 0–1.
        :param noise_levels: Five fog-noise-amount anchors mapped to scales 0–1.
        :returns: Fogged uint8 image.
        """
        intensity = self._interpolate(scale, intensity_levels)
        noise_amount = self._interpolate(scale, noise_levels)
        image = np.array(image, dtype=np.uint8)
        fog_overlay = np.full_like(image, 255, dtype=np.uint8)
        noise = (
            np.random.normal(scale=noise_amount * 255, size=image.shape)
            .clip(0, 255)
            .astype(np.uint8)
        )
        fog_overlay = cv2.addWeighted(fog_overlay, 1 - noise_amount, noise, noise_amount, 0)
        return cv2.addWeighted(image, 1 - intensity, fog_overlay, intensity, 0)

    def snow_filter(
        self,
        scale: float,
        image: np.ndarray,
        intensity_levels: tuple[float, ...] = (0.15, 0.22, 0.3, 0.45, 0.6),
        **_: Any,
    ) -> np.ndarray:
        """Overlay a snow texture from ``snow.png`` with alpha blending.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param intensity_levels: Five overlay-intensity anchors mapped to scales 0–1.
        :returns: Snow-overlaid uint8 image with reduced saturation.
        :raises FileNotFoundError: If ``snow.png`` is not found in the assets directory.
        :raises IOError: If ``snow.png`` cannot be decoded by OpenCV.
        """
        intensity = self._interpolate(scale, intensity_levels)
        if not self._frost_path.exists():
            raise FileNotFoundError(f"Could not find `snow.png`. Expected at `{self._frost_path}`.")
        frost_overlay = cv2.imread(str(self._frost_path), cv2.IMREAD_UNCHANGED)
        if frost_overlay is None:
            raise IOError(f"File `{self._frost_path}` exists but could not be decoded by OpenCV.")
        frost_overlay_resized = cv2.resize(frost_overlay, (image.shape[1], image.shape[0]))
        bgr = frost_overlay_resized[:, :, :3]
        alpha = frost_overlay_resized[:, :, 3] / 255.0
        frosted = np.clip(
            (1 - intensity * alpha[:, :, np.newaxis]) * image + intensity * bgr, 0, 255
        ).astype(np.uint8)
        hsv = cv2.cvtColor(frosted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.8
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def contrast(
        self,
        scale: float,
        image: np.ndarray,
        factor_levels: tuple[float, ...] = (1.1, 1.2, 1.3, 1.5, 1.7),
        **_: Any,
    ) -> np.ndarray:
        """Increase image contrast by scaling pixel values around the mid-grey point.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param factor_levels: Five contrast-factor anchors mapped to scales 0–1.
        :returns: Contrast-enhanced uint8 image.
        """
        contrast_factor = self._interpolate(scale, factor_levels)
        mid_level = 127.5
        return np.clip(mid_level + (image - mid_level) * contrast_factor, 0, 255).astype(np.uint8)

    def elastic(
        self,
        scale: float,
        image: np.ndarray,
        alpha_levels: tuple[int, ...] = (2, 3, 5, 7, 10),
        sigma_levels: tuple[float, ...] = (0.4, 0.75, 0.9, 1.2, 1.5),
        **_: Any,
    ) -> np.ndarray:
        """Apply elastic deformation via smooth random displacement fields.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param alpha_levels: Five displacement-amplitude anchors mapped to scales 0–1.
        :param sigma_levels: Five Gaussian-smoothing-sigma anchors mapped to scales 0–1.
        :returns: Elastically deformed uint8 image.
        """
        alpha = self._interpolate(scale, alpha_levels)
        sigma = self._interpolate(scale, sigma_levels)
        dx = cv2.GaussianBlur(np.random.uniform(-1, 1, image.shape[:2]) * alpha, (0, 0), sigma)
        dy = cv2.GaussianBlur(np.random.uniform(-1, 1, image.shape[:2]) * alpha, (0, 0), sigma)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        return cv2.remap(
            image,
            (x + dx).astype(np.float32),
            (y + dy).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def cutout_filter_with_bbox(
        self,
        scale: float,
        image: np.ndarray,
        bboxes: list[list[int]],
        patch_count_levels: tuple[int, ...] = (1, 2, 4, 6, 10),
        coverage_levels: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.33),
        **_: Any,
    ) -> np.ndarray:
        """Black-out random rectangular patches while respecting a per-bbox coverage limit.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 image.
        :param bboxes: Ground-truth bounding boxes as ``[xmin, ymin, xmax, ymax]`` lists.
        :param patch_count_levels: Five target-patch-count anchors mapped to scales 0–1.
        :param coverage_levels: Five max-coverage-fraction anchors mapped to scales 0–1.
        :returns: Image with black rectangular cutouts applied.
        """
        image = image.copy()
        h, w, _ = image.shape
        target_count = int(self._interpolate(scale, patch_count_levels))
        target_coverage = self._interpolate(scale, coverage_levels)

        bbox_masks = [np.zeros((b[3] - b[1], b[2] - b[0])) for b in bboxes]
        attempts = 0
        patches_applied = 0

        while patches_applied < target_count and attempts < target_count * 5:
            attempts += 1
            patch_h = np.random.randint(int(h * 0.05), int(h * 0.2) + 1)
            patch_w = np.random.randint(int(w * 0.05), int(w * 0.2) + 1)
            if h - patch_h <= 0 or w - patch_w <= 0:
                continue
            x = np.random.randint(0, h - patch_h)
            y = np.random.randint(0, w - patch_w)

            valid_patch = True
            temp_mask_updates = []
            for i, box in enumerate(bboxes):
                bx_min, by_min, bx_max, by_max = box
                box_area = (bx_max - bx_min) * (by_max - by_min)
                if box_area <= 0:
                    continue
                ir1, ir2 = max(x, by_min), min(x + patch_h, by_max)
                ic1, ic2 = max(y, bx_min), min(y + patch_w, bx_max)
                if ir1 < ir2 and ic1 < ic2:
                    lr1, lr2 = ir1 - by_min, ir2 - by_min
                    lc1, lc2 = ic1 - bx_min, ic2 - bx_min
                    new_pixels = (ir2 - ir1) * (ic2 - ic1) - np.sum(bbox_masks[i][lr1:lr2, lc1:lc2])
                    if (np.sum(bbox_masks[i]) + new_pixels) / box_area > target_coverage:
                        valid_patch = False
                        break
                    temp_mask_updates.append((i, slice(lr1, lr2), slice(lc1, lc2)))

            if valid_patch:
                image[x : x + patch_h, y : y + patch_w, :] = 0
                for idx, slc_row, slc_col in temp_mask_updates:
                    bbox_masks[idx][slc_row, slc_col] = 1
                patches_applied += 1

        return image

    @staticmethod
    def false_color_filter(scale: float, image: np.ndarray, **_: Any) -> np.ndarray:
        """Shift the hue channel to produce false-colour artefacts.

        :param scale: Severity in [0.0, 1.0]; maps linearly to a hue shift of 0–180°.
        :param image: Input uint8 RGB image.
        :returns: Hue-shifted uint8 RGB image.
        """
        shift = int(scale * 180)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def grayscale_filter(
        self,
        scale: float,
        image: np.ndarray,
        severity_levels: tuple[float, ...] = (0.1, 0.2, 0.35, 0.55, 0.85),
        **_: Any,
    ) -> np.ndarray:
        """Blend the image towards greyscale by the given severity.

        :param scale: Severity in [0.0, 1.0].
        :param image: Input uint8 RGB image.
        :param severity_levels: Five greyscale-blend-weight anchors mapped to scales 0–1.
        :returns: Partially desaturated uint8 image.
        """
        severity = self._interpolate(scale, severity_levels)
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(image, 1 - severity, gray, severity, 0)

    def synthesize(self, z: Any) -> Any:
        raise NotImplementedError("Synthesize not implemented.")
