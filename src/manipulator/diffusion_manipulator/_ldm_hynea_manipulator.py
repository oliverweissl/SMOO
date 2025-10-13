import logging
from typing import Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel
from torch import Tensor, nn

from . import DiffusionCandidateList
from ._diffusion_manipulator import DiffusionManipulator
from ._load_models import load_ldm_celebhq
from ._utils import prepare_cuda
from .hypernets import UNet2DHyperNet


class LDMHyNeAManipulator(DiffusionManipulator):
    """A trainer class for the LDM ControlNet."""

    _device: torch.device

    """Models used."""
    _vae: nn.Module
    _model: UNet2DModel
    _scheduler: DDIMScheduler
    _control_net: UNet2DHyperNet

    def __init__(
        self,
        control_shape: tuple[int, ...],
        batch_size: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the LDM ControlNet Manipulator.

        :param control_shape: Shape of the control signal.
        :param batch_size: Batch size (0 means all - Default).
        :param device: Device to use for compute.
        """
        self._device = prepare_cuda(device, True)
        self._model, self._vae, self._scheduler = load_ldm_celebhq(device=self._device)
        self._batch_size = batch_size

        self._control_net = UNet2DHyperNet(
            model=self._model, scheduler=self._scheduler, control_shape=control_shape
        )

    def manipulate(self, candidates: DiffusionCandidateList, **kwargs) -> Tensor:
        """
        Manipulate inputs with their respective control signals.

        :param candidates: The candidates to manipulate.
        :param kwargs: Additional KW-Args, use `timesteps: int` to modify default 50 diffusion steps.
        :return: The sampled outputs.
        """
        self._control_net.set_candidates(kwargs.get("timesteps", 50))
        xs = []
        for c in candidates:
            x = self._control_net.forward(x=c.xt[0], control=c.control)
            xs.append(x)
        return torch.cat(xs, dim=0)

    def get_diff_steps(
        self, class_labels: list[int], n_steps: int = 50, x_0: Optional[Tensor] = None
    ) -> tuple[Tensor, None]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param class_labels: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising.
        :param x_0: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and None as there are no classes here.
        """
        batch_size = len(class_labels)

        x_cur = (
            x_0.to(self._device)
            if x_0 is not None
            else torch.randn(
                batch_size,
                self._model.in_channels,
                self._model.sample_size,
                self._model.sample_size,
                device=self._device,
            )
        )
        xs = torch.empty(
            n_steps + 1,
            *x_cur.shape,
            device=self._device,
        )
        xs[0] = x_cur

        self._scheduler.set_timesteps(num_inference_steps=n_steps)
        for i, t in enumerate(self._scheduler.timesteps):
            with torch.no_grad():
                residual = self._model(x_cur, t)["sample"]

            x_cur = self._scheduler.step(residual, t, x_cur, eta=0.0)["prev_sample"]
            xs[i + 1] = x_cur

        return xs.detach(), None

    def get_images(self, z: Tensor) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :return: The decoded image, color-range [0,1].
        """
        logging.info("Sampling Images from denoised Latents.")
        if z.ndim == 3:  # Ensure batch dimension is present.
            z = z.unsqueeze(0)

        chunks = (
            (z.size(0) + self._batch_size - 1) // self._batch_size
            if self._batch_size > 0
            else z.size(0)
        )
        decoded = []
        for z_chunk in torch.chunk(z, chunks, dim=0):
            with torch.enable_grad():
                image = self._vae.decode(z_chunk)
                image_processed = image.cpu().permute(0, 2, 3, 1)
                decoded.append((image_processed + 1.0) * 127.5)
        return torch.cat(decoded, dim=0)
