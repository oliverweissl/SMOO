import logging
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from ._diffusion_candidate import DiffusionCandidateList
from ._diffusion_manipulator import DiffusionManipulator
from ._internal.models.sit import SiT
from ._load_models import load_default_sit
from ._utils import prepare_cuda


class SiTManipulator(DiffusionManipulator):
    """A Manipulator made for REPA-E trained SiT diffusion models."""

    _device: torch.device

    """Models used."""
    _vae: nn.Module
    _model: SiT

    """Model specific parameters."""
    _batch_size: int

    # Loaded from SiT
    _latent_size: int
    _in_channels: int
    _latents_scale: Tensor
    _latents_bias: Tensor

    """Sampling and Manipulation."""
    _interpolation_strategy: str
    _cfg: float
    _epsilon: float = 1e-6  # Precision constant

    """Auxiliary lambdas for easy callings."""
    _embed_y: Callable[[list[int]], Tensor]
    _embed_t: Callable[[list[Tensor]], Tensor]
    _manip_function: Callable[..., DiffusionCandidateList]

    """Optimization caches."""
    _null_embedding_cache: Optional[Tensor] = None
    _pre_allocated_buffers: dict[tuple[int, ...], Tensor]

    def __init__(
        self,
        model_file: str,
        cfg_scale: float = 1.5,
        batch_size: int = 0,
        manipulation_strategy: str = "new",
        interpolation_strategy: str = "linear",
        device: Union[torch.device, None] = None,
        require_grad: bool = False,
    ) -> None:
        """
        Initialize the manipulator based on REPA-E diffusion models.

        :param model_file: Path to the model file.
        :param cfg_scale: Classifier free guidance scale for conditions in the sampling.
        :param batch_size: Batch size to use for generation of samples (0 means all elements get taken).
        :param manipulation_strategy: Manipulation strategy to use.
        :param interpolation_strategy: Interpolation strategy to use.
        :param device: CUDA device to use if available.
        :param require_grad: Whether to enable gradients for training operations.
        :raises ValueError: If manipulation_strategy is not supported.
        """
        self.require_grad = require_grad
        self._device = prepare_cuda(device, require_grad)
        self._batch_size = batch_size

        self._cfg = cfg_scale

        """Loading models and other variables as locals."""
        loaded = load_default_sit(model_file=model_file, device=device)
        for name, value in vars(loaded).items():
            if not name.startswith("__"):
                setattr(self, f"_{name}", value)

        """Define Embedding lambdas"""
        if require_grad:
            self._embed_y = lambda y: self._model.y_embedder(
                torch.tensor(y, device=self._device), self._model.training
            )
        else:
            self._embed_y = lambda y: self._model.y_embedder(
                torch.tensor(y, device=self._device), self._model.training
            ).detach()

        """Pre-cache null embedding for CFG."""
        self._null_embedding_cache = None

        self._interpolation_strategy = interpolation_strategy
        self._pre_allocated_buffers = {}
        if manipulation_strategy == "base":
            self._manip_function = self._manip_aggregate_base
        elif manipulation_strategy == "new":
            self._manip_function = self._manip_aggregate_new
        else:
            raise ValueError(f"Strategy '{manipulation_strategy}' is not supported.'")

    def manipulate(
        self,
        candidates: DiffusionCandidateList,
        **kwargs,
    ) -> Union[Tensor, tuple[Tensor, DiffusionCandidateList]]:
        """
        Manipulate the diffusion processes of candidates.

        :param candidates: Candidates to manipulate.
        :param kwargs: Additional KW-Args, needs `weights_x: Tensor` and `weights_y: Tensor` to work.
        :returns: The resulting diffusion result.
        :raises IndexError: If candidates have no origin or targets.
        """
        weights_x = kwargs["weights_x"].to(self._device)
        weights_y = kwargs["weights_y"].to(self._device)

        logging.info(f"Manipulating {len(candidates)} candidates.")
        t_steps = torch.linspace(
            1, 0, len(candidates[0].xt), device=self._device
        )  # Define step range.

        if weights_x.ndim > 1:  # Check if we have a batch dimension in the weights.
            batch_size = weights_x.shape[0]
        else:  # If not make a singleton batch dimension for compatability.
            batch_size = 1
            weights_x = weights_x.unsqueeze(0)
            weights_y = weights_y.unsqueeze(0)

        """Convert into batches with optimized expansion."""
        if candidates.target:
            target_xt = candidates.target[0].xt
            target_emb = candidates.target[0].class_embedding
        else:
            raise IndexError("Candidates must have target")
        if candidates.origin:
            origin_xt = candidates.origin[0].xt
            origin_emb = candidates.origin[0].class_embedding
        else:
            raise IndexError("Candidates must have origins")

        xt_target = target_xt.unsqueeze(0).expand(batch_size, *target_xt.shape).contiguous()
        xt_origin = origin_xt.unsqueeze(0).expand(batch_size, *origin_xt.shape).contiguous()
        y_target = target_emb.unsqueeze(0).expand(batch_size, *target_emb.shape).contiguous()
        y_origin = origin_emb.unsqueeze(0).expand(batch_size, *origin_emb.shape).contiguous()

        """The manipulation process."""
        manip_cand = self._manip_function(
            t_steps=t_steps,
            origin=(xt_origin, y_origin),
            target=(xt_target, y_target),
            weights_x=weights_x,
            weights_y=weights_y,
        )

        # Explicit cleanup with immediate cache clearing
        del xt_target, xt_origin, y_origin, y_target, target_xt, origin_xt, target_emb, origin_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        """Return a candidate with the manipulation history if wanted."""
        return (
            (manip_cand.xts[:, -1, ...], manip_cand)
            if kwargs.get("return_manipulation_history", False)
            else manip_cand.xts[:, -1, ...]
        )

    def _manip_aggregate_new(
        self,
        *,
        t_steps: Tensor,
        origin: tuple[Tensor, Tensor],
        target: tuple[Tensor, Tensor],
        weights_x: Tensor,
        weights_y: Tensor,
    ) -> DiffusionCandidateList:
        """
        Manipulate latent vector of diffusion processes and aggregate them on the new candidate.

        :param t_steps: Time steps to manipulate.
        :param origin: Tuple of origin diffusion process and embedding.
        :param target: Tuple of target diffusion process and embedding.
        :param weights_x: Weights to manipulate diffusion process.
        :param weights_y: Weights to manipulate class embeddings.
        :returns: The resulting diffusion candidate.
        """
        xt_origin, y_origin = origin
        xt_target, y_target = target

        # Pre-compute interpolated embeddings
        y_manip = self.interpolate(y_origin, y_target, weights_y, dim=(1,)).float()

        # Pre-allocate template with better memory layout
        buffer_key = xt_origin.shape
        if buffer_key not in self._pre_allocated_buffers:
            self._pre_allocated_buffers[buffer_key] = torch.empty_like(xt_origin)
        xt_template = self._pre_allocated_buffers[buffer_key]
        xt_template.zero_()

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_manip = self.interpolate(
                xt_origin[:, i, ...], xt_target[:, i, ...], weights_x[:, i], dim=(1,)
            ).float()
            x_manip.add_(xt_template[:, i, ...]).mul_(0.5)
            x_cur = self._sample(t=t_cur, x=x_manip, y=y_manip[:, i, ...], step=t_next - t_cur)
            xt_template[:, i + 1, ...] = x_cur
        # Transposing from Batch x DiffSteps x Z -> DiffSteps x Batch x Z
        diff_candidates = DiffusionCandidateList.from_diffusion_output(
            xs=xt_template.transpose(0, 1).detach(), emb=y_manip.detach(), separate_candidates=False
        )
        return diff_candidates

    def _manip_aggregate_base(
        self,
        *,
        t_steps: Tensor,
        origin: tuple[Tensor, Tensor],
        target: tuple[Tensor, Tensor],
        weights_x: Tensor,
        weights_y: Tensor,
    ) -> DiffusionCandidateList:
        """
        Manipulate latent vector of diffusion processes and aggregate them on the base candidate.

        :param t_steps: Time steps to manipulate.
        :param origin: Tuple of origin diffusion process and embedding.
        :param target: Tuple of target diffusion process and embedding.
        :param weights_x: Weights to manipulate diffusion process.
        :param weights_y: Weights to manipulate class embeddings.
        :returns: The resulting diffusion candidate.
        """
        logging.warning(
            "Attention: This manipulation is very minimal as it depends on the origin image!"
        )
        xt_origin, y_origin = origin
        xt_target, y_target = target

        """Do the manipulations on batches with optimized memory usage."""
        y_manip = self.interpolate(y_origin, y_target, weights_y, dim=(1,)).float()

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_manip = self.interpolate(
                xt_origin[:, i, ...], xt_target[:, i, ...], weights_x[:, i], dim=(1,)
            ).float()
            x_cur = self._sample(t=t_cur, x=x_manip, y=y_manip[:, i, ...], step=t_next - t_cur)
            xt_target[:, i + 1, ...] = x_cur
            del x_manip, x_cur

        # Transposing from Batch x DiffSteps x Z -> DiffSteps x Batch x Z
        diff_candidates = DiffusionCandidateList.from_diffusion_output(
            xs=xt_target.transpose(0, 1).detach(), emb=y_manip.detach(), separate_candidates=False
        )
        return diff_candidates

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

        with torch.enable_grad() if self.require_grad else torch.no_grad():
            t_curr = torch.full(size=(y.size(0),), fill_value=t.item(), device=self._device)
            if cond:
                model_input = x.repeat(2, *([1] * (x.ndim - 1)))
                # Use cached null embedding if available
                if (
                    self._null_embedding_cache is None
                    or self._null_embedding_cache.shape[0] != y.shape[0]
                ):
                    null_embedding_cache = self._embed_y([1000] * y.shape[0])
                else:
                    null_embedding_cache = self._null_embedding_cache
                y_curr = torch.cat((y, null_embedding_cache), dim=0)
                t_curr = t_curr.repeat(2, *([1] * (t_curr.ndim - 1)))
            else:
                model_input, y_curr = x, y
            d_cur = self._model.partial_inference(
                x=model_input, t=t_curr, y=y_curr, require_grad=self.require_grad
            )

        if cond:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + self._cfg * (d_cur_cond - d_cur_uncond)

        return x + step * d_cur

    def get_diff_steps(
        self, diff_input: list[int], n_steps: Optional[int] = None, x_0: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param diff_input: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising (Default=50).
        :param x_0: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and the class embedding.
        """
        batch_size = len(diff_input)
        n_steps = n_steps or 50

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
            batch_size,
            self._in_channels,
            self._latent_size,
            self._latent_size,
            device=self._device,
        )
        xs[0] = x_cur  # Store the initial Noise.

        # Optimized diffusion loop with in-place updates
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = self._sample(t=t_cur, x=x_cur, y=y_cur, step=t_next - t_cur)
            xs[i + 1] = x_cur

        return xs.detach(), y_cur

    def get_images(self, z: Tensor) -> Tensor:
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
            with torch.enable_grad() if self.require_grad else torch.no_grad():
                decoded_latents = (z_chunk / self._latents_scale) + self._latents_bias
                element = self._vae.decode(decoded_latents).sample
                element = torch.clamp(element.mul_(0.5).add_(0.5), 0.0, 1.0)
                decoded.append(element)
        return torch.cat(decoded, dim=0)

    @staticmethod
    def linear(p: Tensor, q: Tensor, weight: Tensor) -> Tensor:
        """
        Weights for linear interpolation.

        :param p: The first tensor.
        :param q: The second tensor.
        :param weight: The weight for the interpolation.
        :return: The interpolated tensor.
        """
        wp, wq = 1 - weight, weight
        return wp * p + wq * q

    def interpolate(
        self, p: Tensor, q: Tensor, weight: Tensor, dim: Optional[tuple[int, ...]] = None
    ) -> Tensor:
        """
        Implement optimized interpolation for Tensors.

        :param p: The first tensor.
        :param q: The second tensor.
        :param weight: The weight for the interpolation.
        :param dim: The dimensions to do operations across.
        :return: The interpolated tensor.
        :raises ValueError: if interpolation strategy is not known.
        """
        if weight.ndim == 0:
            weight = weight.view(1)

        target_ndim = p.ndim
        current_ndim = weight.ndim
        if current_ndim < target_ndim:
            shape_ext = [1] * (target_ndim - current_ndim)
            weight = weight.view(*weight.shape, *shape_ext)

        # Branch prediction optimization - most common case first
        if self._interpolation_strategy == "linear":
            return self.linear(p, q, weight)
        elif self._interpolation_strategy == "slerp":
            return self.slerp(p, q, weight, dim)
        else:
            raise ValueError(f"Unknown interpolation strategy {self._interpolation_strategy}")

    def slerp(self, p: Tensor, q: Tensor, weight: Tensor, dim: Optional[tuple[int, ...]]) -> Tensor:
        """
        Optimized spherical linear interpolation.

        :param p: The first tensor.
        :param q: The second tensor.
        :param weight: The weight for the interpolation.
        :param dim: The dimensions to do operations across.
        :return: The interpolated tensor.
        """
        p_norm_val = torch.linalg.norm(p, dim=dim, keepdim=True).clamp_min(self._epsilon)
        q_norm_val = torch.linalg.norm(q, dim=dim, keepdim=True).clamp_min(self._epsilon)

        p_norm = p / p_norm_val
        q_norm = q / q_norm_val

        dot = torch.sum(p_norm * q_norm, dim=dim, keepdim=True).clamp(
            -1.0 + self._epsilon, 1.0 - self._epsilon
        )

        abs_dot = torch.abs(dot)
        fallback_threshold = 1.0 - (5 * self._epsilon)
        if torch.all(abs_dot > fallback_threshold):
            return self.linear(p, q, weight)

        omega = torch.arccos(dot)
        sin_omega = torch.sin(omega).clamp_min(self._epsilon)
        omega_w = omega * weight

        wp = torch.sin(omega - omega_w) / sin_omega
        wq = torch.sin(omega_w) / sin_omega
        return wp * p + wq * q
