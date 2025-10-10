import gc
import logging
from typing import Optional

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn

from .. import Manipulator
from ._internal.models.sit import SiT
from ._load_models import load_default_sit
from ._utils import prepare_cuda
from .hypernets import SiTHyperNet


class SitHyNeAManipulator(Manipulator):
    """A trainer class for the ControlNet."""

    _device: torch.device

    """Models used."""
    _vae: nn.Module
    _model: SiT
    _control_net: SiTHyperNet

    # Loaded from SiT
    _latent_size: int
    _in_channels: int
    _latents_scale: Tensor
    _latents_bias: Tensor

    def __init__(
        self,
        model_file: str,
        control_shape: tuple[int, ...],
        cfg_scale: float = 1.5,
        batch_size: int = 0,
        device: Optional[torch.device] = None,
        diffusion_steps: int = 50,
    ) -> None:
        """
        Initialize the manipulator based on REPA-E diffusion models.

        :param model_file: Model file to load weights from.
        :param control_shape: The shape of the control map.
        :param cfg_scale: Classifier free guidance scale for conditions in the sampling.
        :param batch_size: Batch size of operations (Default=0, takes all images at once).
        :param device: CUDA device to use if available.
        :param diffusion_steps: The number of diffusion steps to use.
        """
        self._device = prepare_cuda(device, True)
        self._batch_size = batch_size

        self._cfg = cfg_scale

        """Loading models and other variables as locals."""
        loaded = load_default_sit(model_file=model_file, device=device)
        for name, value in vars(loaded).items():
            if not name.startswith("__"):
                setattr(self, f"_{name}", value)

        """Define Embedding lambdas"""
        self._embed_y = lambda y: self._model.y_embedder(
            torch.tensor(y, device=self._device), self._model.training
        )

        """ControlNet stuff."""
        self._control_shape = control_shape
        self.make_fresh_hyper_net()

        self.noise_loss = nn.MSELoss()
        self.scheduler = DDPMScheduler(
            num_train_timesteps=50, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"
        )
        self._n_steps = diffusion_steps

    def make_fresh_hyper_net(self) -> None:
        """Create a new ControlNet for the current model. ATTENTION: Deletes old one if exists!."""
        if hasattr(self, "_control_net"):
            del self._control_net
            gc.collect()
            torch.cuda.empty_cache()
        self._control_net = SiTHyperNet(self._model, self._control_shape)
        self._control_net.to(self._device)

    def manipulate(self, x: Tensor, y: list[int], c: Tensor) -> Tensor:
        """
        Manipulate inputs with their respective control signals.

        :param x: Tensor of shape `(B, C, H, W)`.
        :param y: The labels for each X (len=B).
        :param c: The control signals for each X.
        :return: The sampled outputs.
        """
        y_cur = self._embed_y(y)
        y_null = self._embed_y([1000] * y_cur.shape[0])
        x = self._control_net.forward(
            x=x, y=y_cur, control=c, cfg=self._cfg, guidance_bounds=(0.0, 1.0), y_null=y_null
        )
        return x

    @property
    def control_net(self) -> SiTHyperNet:
        """
        Get the controlnet used.

        :return: The controlnet used.
        """
        return self._control_net

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing.

        :param enable: Whether to enable gradient checkpointing.
        """
        if enable:
            self._control_net.gradient_checkpointing_enable()
        else:
            self._control_net.gradient_checkpointing_disable()

    def get_image(self, z: Tensor) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :return: The decoded image.
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
                decoded_latents = (z_chunk / self._latents_scale) + self._latents_bias
                element = self._vae.decode(decoded_latents).sample
                element = torch.clamp(element.mul_(0.5).add_(0.5), 0.0, 1.0)
                decoded.append(element)
        return torch.cat(decoded, dim=0)
