import gc
import logging
from typing import Any, Callable, Optional

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
    _hyper_net: UNet2DHyperNet

    def __init__(
        self,
        control_shape: tuple[int, ...],
        batch_size: int = 0,
        diffusion_steps: int = 50,
        device: Optional[torch.device] = None,
        model_loader: Callable[
            [torch.device], tuple[UNet2DModel, nn.Module, DDIMScheduler]
        ] = load_ldm_celebhq,
        guidance_scale: float = 1.0,
    ) -> None:
        """
        Initialize the LDM ControlNet Manipulator.

        :param control_shape: Shape of the control signal.
        :param batch_size: Batch size (0 means all - Default).
        :param diffusion_steps: Diffusion steps to take in denoising.
        :param device: Device to use for compute.
        :param model_loader: A loader function that provides the models for the manipulator.
        :param guidance_scale: Classifier-free guidance scale (1.0=no CFG, >1.0 enables CFG).
        """
        self.control_shape = control_shape

        self._device = prepare_cuda(device, True)
        self._model, self._vae, self._scheduler = model_loader(self._device)
        self._dtype = self._model.dtype
        self._batch_size = batch_size
        self._diffusion_steps = diffusion_steps
        self._guidance_scale = guidance_scale

        for p in self._vae.parameters():
            p.requires_grad_(False)  # Freeze vae parameters
        self.make_fresh_hyper_net()

    def make_fresh_hyper_net(self) -> None:
        """Create a new ControlNet for the current model. ATTENTION: Deletes old one if exists!."""
        if hasattr(self, "_hyper_net"):
            del self._hyper_net
            gc.collect()
            torch.cuda.empty_cache()
        self._hyper_net = UNet2DHyperNet(
            model=self._model,
            scheduler=self._scheduler,
            control_shape=self.control_shape,
            guidance_scale=self._guidance_scale,
        )
        self._hyper_net.to(self._device, dtype=self._dtype)

    def manipulate(self, candidates: DiffusionCandidateList, **kwargs) -> Tensor:
        """
        Manipulate inputs with their respective control signals.

        :param candidates: The candidates to manipulate.
        :param kwargs: Additional KW-Args, use `timesteps: int` to modify default 50 diffusion steps.
        :return: The sampled outputs.
        """
        xs = []

        model_kwargs = kwargs.get("model_kwargs", {})
        if "encoder_hidden_states" in model_kwargs:
            model_kwargs = {
                **model_kwargs,
                "encoder_hidden_states": model_kwargs["encoder_hidden_states"].detach(),
            }
        if "added_cond_kwargs" in model_kwargs:
            model_kwargs = {
                **model_kwargs,
                "added_cond_kwargs": {
                    k: v.detach() if isinstance(v, torch.Tensor) else v
                    for k, v in model_kwargs["added_cond_kwargs"].items()
                },
            }

        for c in candidates:
            assert isinstance(
                c.control, Tensor
            ), "Error: Control is needed for this type of manipulator."
            x = self._hyper_net.forward(
                x=c.xt[0].unsqueeze(0).to(self._device, self._dtype),
                control=c.control.to(self._device, self._dtype),
                timesteps=c.num_diffusion_steps or self._diffusion_steps,
                start_step=c.start_step_idx or 0,
                **model_kwargs,
            )
            xs.append(x)
        return torch.cat(xs, dim=0)

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing.

        :param enable: Whether to enable gradient checkpointing.
        """
        self._hyper_net.use_checkpoints = enable

    def get_diff_steps(
        self,
        diff_input: list[int],
        n_steps: Optional[int] = None,
        x_0: Optional[Tensor] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        start_step: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param diff_input: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising.
        :param x_0: Optional starting latent vector if sampled differently.
        :param model_kwargs: Optional model kwargs.
        :param start_step: Which timestep to start denoising from (0 = denoise all steps).
        :returns: A list of latent vectors through denoising and empty tensor as there are no classes here.
        """
        batch_size = len(diff_input)
        n_steps = n_steps or self._diffusion_steps

        x_cur = (
            x_0.to(self._device, self._dtype)
            if x_0 is not None
            else torch.randn(
                batch_size,
                self._model.config["in_channels"],
                self._model.sample_size,
                self._model.sample_size,
                device=self._device,
                dtype=self._dtype,
            )
        )

        self._scheduler.set_timesteps(num_inference_steps=n_steps)
        timesteps = self._scheduler.timesteps[start_step:]

        xs = torch.empty(
            len(timesteps) + 1,
            *x_cur.shape,
            device=self._device,
            dtype=self._dtype,
        )
        xs[0] = x_cur

        for i, t in enumerate(timesteps):
            # Scale the model input according to the scheduler (critical for DDIM/SDXL)
            x_scaled = self._scheduler.scale_model_input(x_cur, t)
            residual, *_ = self._model(
                x_scaled, t, return_dict=False, **model_kwargs if model_kwargs else {}
            )
            x_cur, *_ = self._scheduler.step(residual, t, x_cur, eta=0.0, return_dict=False)
            xs[i + 1] = x_cur

        return xs.detach(), torch.empty(1, device=self._device, dtype=self._dtype)

    def synthesize(self, z: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :param eps: The epsilon value to avoid gradient instabilities.
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
            if hasattr(self._vae, "config") and hasattr(self._vae.config, "scaling_factor"):
                z_chunk = z_chunk / self._vae.config.scaling_factor
            image, *_ = self._vae.decode(z_chunk, return_dict=False)
            image = (image * 0.5 + 0.5).clamp(0.0 + eps, 1.0 - eps)
            decoded.append(image)
        return torch.cat(decoded, dim=0)

    @property
    def hyper_net(self) -> UNet2DHyperNet:
        """
        Get the HyperNet used.

        :return: The HyperNet used.
        """
        return self._hyper_net
