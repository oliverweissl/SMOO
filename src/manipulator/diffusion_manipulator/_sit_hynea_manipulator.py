import gc
import logging
from typing import Optional

import torch
from torch import Tensor, nn

from ._diffusion_candidate import DiffusionCandidateList
from ._diffusion_manipulator import DiffusionManipulator
from ._internal.models.sit import SiT
from ._load_models import load_default_sit
from ._utils import prepare_cuda
from .hypernets import SiTHyperNet


class SitHyNeAManipulator(DiffusionManipulator):
    """A trainer class for the ControlNet."""

    _device: torch.device

    """Models used."""
    _vae: nn.Module
    _model: SiT
    _hyper_net: SiTHyperNet

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
        for p in self._vae.parameters():
            p.requires_grad_(False)  # Freeze vae parameters

        """Define Embedding lambdas"""
        self._embed_y = lambda y: self._model.y_embedder(
            torch.tensor(y, device=self._device), self._model.training
        )

        """ControlNet stuff."""
        self.control_shape = control_shape
        self.make_fresh_hyper_net()
        self._n_steps = diffusion_steps

    def make_fresh_hyper_net(self) -> None:
        """Create a new ControlNet for the current model. ATTENTION: Deletes old one if exists!."""
        if hasattr(self, "_hyper_net"):
            del self._hyper_net
            gc.collect()
            torch.cuda.empty_cache()
        self._hyper_net = SiTHyperNet(self._model, self.control_shape)
        self._hyper_net.to(self._device)

    def get_diff_steps(
        self, diff_input: list[int], n_steps: Optional[int] = None, x_0: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param diff_input: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising.
        :param x_0: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and the class embedding.
        """
        batch_size = len(diff_input)
        n_steps = n_steps or self._n_steps

        x_cur = (
            x_0.to(self._device)
            if x_0 is not None
            else torch.randn(
                batch_size,
                self._in_channels,
                self._latent_size,
                self._latent_size,
                device=self._device,
            )
        )

        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)
        y_cur = self._embed_y(diff_input)

        xs = torch.empty(
            n_steps + 1,
            *x_cur.shape,
            device=self._device,
        )
        xs[0] = x_cur  # Store the initial Noise.
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = self._sample(t=t_cur, x=x_cur, y=y_cur, step=t_next - t_cur)
            xs[i + 1] = x_cur

        return xs.detach(), y_cur

    def manipulate(self, candidates: DiffusionCandidateList, **kwargs) -> Tensor:
        """
        Manipulate inputs with their respective control signals.

        :param candidates: The candidates to manipulate.
        :param kwargs: Additional KW-Args.
        :return: The sampled outputs.
        """
        xs = []
        # TODO: could be run all together instead of for loop ?? -> maybe too much memory usage tho.
        for c in candidates:
            y_cur = self._embed_y([c.y])
            y_null = self._embed_y([1000] * y_cur.shape[0])
            x = self._hyper_net.forward(
                x=c.xt[0].unsqueeze(0),  # Here we add a pseudo-batch dimension.
                y=y_cur,
                control=c.control,
                cfg=self._cfg,
                guidance_bounds=(0.0, 1.0),
                y_null=y_null,
            )
            xs.append(x)
        return torch.cat(xs, dim=0)

    @property
    def hyper_net(self) -> SiTHyperNet:
        """
        Get the HyperNet used.

        :return: The HyperNet used.
        """
        return self._hyper_net

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing.

        :param enable: Whether to enable gradient checkpointing.
        """
        self._hyper_net.use_checkpoints = enable

    def get_images(self, z: Tensor, eps: float = 1e-6) -> Tensor:
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
            decoded_latents = (z_chunk / self._latents_scale) + self._latents_bias
            element = self._vae.decode(decoded_latents).sample
            element = (element * 0.5 + 0.5).clamp(0.0 + eps, 1.0 - eps)
            decoded.append(element)
        return torch.cat(decoded, dim=0)

    def _sample(
        self,
        t: Tensor,
        x: Tensor,
        y: Tensor,
        step: float,
        guidance_bounds: tuple[float, float] = (0.0, 1.0),
        **_,
    ) -> Tensor:
        """
        Sampling new outputs based on euler_sampler in REPA-E repo.

        :param t: The current time step.
        :param x: The current state of the diffusion process.
        :param y: The current class embedding of the diffusion process.
        :param step: The step size.
        :param guidance_bounds: Guidance bounds for conditions in the sampling.
        :param _: Unused keyword arguments.
        :returns: The sampled outputs for the current timestep.
        """
        cond = self._cfg > 1.0 and guidance_bounds[1] >= t >= guidance_bounds[0]

        with torch.enable_grad():
            t_curr = torch.full(size=(y.size(0),), fill_value=t.item(), device=self._device)
            if cond:
                model_input = x.repeat(2, *([1] * (x.ndim - 1)))
                null_embedding_cache = self._embed_y([1000] * y.shape[0])

                y_curr = torch.cat((y, null_embedding_cache), dim=0)
                t_curr = t_curr.repeat(2, *([1] * (t_curr.ndim - 1)))
            else:
                model_input, y_curr = x, y
            d_cur = self._model.partial_inference(
                x=model_input, t=t_curr, y=y_curr, require_grad=True
            )

        if cond:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + self._cfg * (d_cur_cond - d_cur_uncond)

        return x + step * d_cur
