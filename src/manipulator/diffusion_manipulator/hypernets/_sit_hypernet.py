import math
from copy import deepcopy
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .._internal.models.sit import SiT


def mean_flat(x: Tensor) -> Tensor:
    """
    Take the mean over all non-batch dimensions.

    :param x: A tensor to average over.
    :returns: The average of the tensor over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class ZeroLinear(nn.Linear):
    """A Zero-initialized Linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initializer the ZeroLinear layer.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param bias: Whether to use a bias.
        """
        super().__init__(in_features, out_features, bias)
        nn.init.zeros_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)


class SiTHyperNet(nn.Module):
    """A ControlNet-like implementation for SiT."""

    _raw_y: bool

    def __init__(
        self,
        sit_model: SiT,
        control_shape: tuple[int, ...],
        train_frac: Optional[float] = None,
        raw_y: bool = False,
    ) -> None:
        """
        Initialize the SiTControlNet.

        :param sit_model: The underlying SiT model to control.
        :param control_shape: The shape of the control tokens to use. Should not include the batch dimension.
        :param train_frac: The fraction of blocks to be trainable (Default: Encoder Depth of SiT.).
        :param raw_y: if the y-Tensors parsed are raw class-conditions (True) or are already embedded (False) [Default: False].
        """
        super().__init__()
        self._raw_y = raw_y
        if train_frac is None:
            self.cutoff = sit_model.encoder_depth
        else:
            assert 0 < train_frac <= 1, "Train fraction must be between 0 and 1."
            self.cutoff = int(len(sit_model.blocks) * train_frac)
            assert self.cutoff > 0, "Train fraction must be greater than 0."

        """Define Control Blocks."""
        self.control_layers = nn.ModuleList(deepcopy(sit_model.blocks[: self.cutoff]))
        module_list = []
        for block in self.control_layers:
            in_features = block.mlp.fc1.in_features
            out_features = block.mlp.fc2.out_features
            module_list.append(ZeroLinear(in_features, out_features))
        self.zero_layers = nn.ModuleList(module_list)

        """Define projection of control input."""
        in_size = sit_model.blocks[0].attn.qkv.in_features
        self.control_embed = ZeroLinear(math.prod(control_shape), in_size)
        self.control_projection = nn.MultiheadAttention(in_size, num_heads=8, batch_first=True)

        """Copy SiT functionality."""
        self.base_model = sit_model
        for param in self.base_model.parameters():
            param.requires_grad_(False)  # Freeze parameters

    def trainable_parameters(self) -> list[nn.Parameter]:
        """
        Parse all trainable parameters in the model.

        :returns: A list of trainable parameters in the model (Control-Layers, Control-Projection, Zero-Layers, Control-Embed).
        """
        return [
            *self.control_layers.parameters(),
            *self.control_projection.parameters(),
            *self.control_embed.parameters(),
            *self.zero_layers.parameters(),
        ]

    def forward_full(
        self,
        x: Tensor,
        y: Tensor,
        control: Tensor,
        cfg: float,
        guidance_bounds: tuple[float, float],
        y_null: Optional[Tensor] = None,
        timesteps: int = 50,
        t_bounds: tuple[float, float] = (1.0, 0.0),
    ) -> Tensor:
        """
        Full denoising process - used for end-to-end training.

        :param x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images UNNORMALIZED).
        :param y: (N) tensor of class labels.
        :param control: (N, *S) tensor of control tokens to use for the forward pass.
        :param cfg: Classifier free guidance scale.
        :param guidance_bounds: For which timesteps guidance should be permitted.
        :param y_null: The null-class tensor if guidance is used.
        :param timesteps: Provide the amount of denoising timesteps.
        :param t_bounds: Provide the time bounds for denoising (start, end).
        :returns: The results of the forward pass.
        """
        device, dtype = x.device, x.dtype
        t_start, t_end = t_bounds
        time_inputs = torch.linspace(t_start, t_end, timesteps + 1, device=device, dtype=dtype)[1:]
        y_embed = self._get_y_embed(y)

        cond = cfg > 1.0 and guidance_bounds[1] >= t_start > t_end >= guidance_bounds[0]
        if cond and y_null is not None:
            x = x.repeat(2, *([1] * (x.ndim - 1)))
            y_embed = torch.cat((y_embed, y_null), dim=0)
            control = control.repeat(2, *([1] * (control.ndim - 1)))

        for t_cur, t_next in zip(time_inputs[:-1], time_inputs[1:]):
            x_initial = x.clone()
            t = t_cur.expand(x.size(0))

            x, _ = self._denoise_step(x, t, y_embed, control, use_checkpoint=True)

            if cond:
                x_cond, x_uncond = x.chunk(2)
                x = x_uncond + cfg * (x_cond - x_uncond)

            x = x_initial + (t_next - t_cur) * x

        return x.chunk(2)[0] if cond else x

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        control: Tensor,
        weighting: str = "uniform",
        path_type: str = "linear",
        time_input: Optional[Tensor] = None,
        noises: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Single denoising step training - used for training individual timesteps.

        :param x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images UNNORMALIZED).
        :param y: (N,) tensor of class labels.
        :param weighting: Weighting scheme for time input (uniform or lognormal).
        :param path_type: Interpolant path (linear or cosine).
        :param control: (N, *S) tensor of control tokens to use for the forward pass.
        :param time_input: Provide an optional tensor of timesteps to use for the forward pass, otherwise sample from a distribution.
        :param noises: Provide an optional tensor of noises to use for the forward pass, otherwise sample from a distribution.
        :returns: The results of the forward pass, including the denoising loss.
        """
        x = self.base_model.bn(x)
        time_input = self._time_input(time_input, x, weighting, path_type)
        time_input = time_input.to(device=x.device, dtype=x.dtype)

        noises = (
            torch.randn_like(x) if noises is None else noises.to(device=x.device, dtype=x.dtype)
        )
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.base_model.interpolant(
            time_input, path_type=path_type
        )
        model_input = alpha_t * x + sigma_t * noises

        y_embed = self._get_y_embed(y)
        pred, x_embedded = self._denoise_step(model_input, time_input.flatten(), y_embed, control)

        model_target = d_alpha_t * x_embedded + d_sigma_t * noises
        denoising_loss = mean_flat((pred - model_target) ** 2)
        return pred, denoising_loss

    @torch.no_grad()
    def inference(self, x: Tensor, t: Tensor, y: Tensor, control: Tensor) -> Tensor:
        """
        Single denoising step inference - used for sampling/generation.

        :param x: The noisy latent representations of images.
        :param t: The timesteps to use for the denoising.
        :param y: The class labels to use for the denoising.
        :param control: The control map to use for the denoising.
        :returns: The one step denoised latent representations of images.
        """
        y_embed = self._get_y_embed(y)
        x, _ = self._denoise_step(x, t, y_embed, control)
        return x

    """Private methods in the ControlNet."""

    def _time_input(
        self, time_input: Optional[Tensor], normalized_x: Tensor, weighting: str, path_type: str
    ) -> Tensor:
        """
        Generates time input for the forward pass.

        :param time_input: The time input tensor, if provided.
        :param normalized_x: The normalized input tensor.
        :param weighting: Weighting scheme for time input (uniform or lognormal).
        :param path_type: Interpolant parth (linear or cosine).
        :returns: The time input tensor.
        :raises NotImplementedError: If the weighting scheme is not supported.
        """
        if time_input is None:
            if weighting == "uniform":
                time_input = torch.rand((normalized_x.shape[0], 1, 1, 1))
            elif weighting == "lognormal":
                rnd_normal = torch.randn((normalized_x.shape[0], 1, 1, 1))
                sigma = rnd_normal.exp()
                if path_type == "linear":
                    time_input = sigma / (1 + sigma)
                elif path_type == "cosine":
                    time_input = 2 / torch.pi * torch.atan(sigma)
            else:
                raise NotImplementedError(f"Weighting scheme {weighting} not implemented.")
        elif time_input.ndim == 1:
            time_input = time_input[:, None, None, None]
        return time_input

    def _get_y_embed(self, y: Tensor) -> Tensor:
        """
        Get y embedding based on configuration.

        :param y: Input y tensor (raw classes or pre-embedded).
        :returns: Y embedding tensor.
        """
        return self.base_model.y_embedder(y, self.training) if self._raw_y else y

    def _get_control_tokens(self, control: Tensor, x: Tensor) -> Tensor:
        """
        Process control input to generate control tokens.

        :param control: (N, *S) tensor of control tokens to use for the forward pass.
        :param x: (N, T, D) tensor of embedded input tokens.
        :returns: (N, T, D) tensor of control tokens.
        """
        control = control.to(device=x.device, dtype=x.dtype)
        control_embed = self.control_embed(control.flatten(1)).unsqueeze(1)
        control_tokens, _ = self.control_projection(x, control_embed, control_embed)
        return control_tokens

    def _denoise_step(
        self, x: Tensor, t: Tensor, y_embed: Tensor, control: Tensor, use_checkpoint: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Core denoising step logic shared by all methods.

        :param x: Input tensor to denoise.
        :param t: Timestep tensor.
        :param y_embed: Y embedding tensor.
        :param control: Control tensor.
        :param use_checkpoint: Whether to use gradient checkpointing.
        :returns: Denoised output tensor and initial embedded x tensor.
        """
        x_embed = self.base_model.x_embedder(x) + self.base_model.pos_embed
        control_tokens = self._get_control_tokens(control, x_embed)
        x_control = x_embed + control_tokens

        t_embed = self.base_model.t_embedder(t)
        c = t_embed + y_embed

        if use_checkpoint:
            controlnet_outputs = checkpoint(
                self._control_forward, x_control, c, use_reentrant=False
            )
        else:
            controlnet_outputs = self._control_forward(x_control, c)

        x = self._backbone_forward(controlnet_outputs, x_embed, c)
        x = self.base_model.final_layer(x, c)
        return self.base_model.unpatchify(x), x_embed

    def _control_forward(self, x_control: Tensor, c: Tensor) -> list[Tensor]:
        """
        The forward pass of the ControlNet only.

        :param x_control: (N, C, H, W) tensor of control conditioned x.
        :param c: The conditioning tensor.
        :returns: A list of outputs; one Tensor for each block in the ControlNet.
        """
        controlnet_outputs = []
        for block, zero_layer in zip(self.control_layers, self.zero_layers):
            x_control = block(x_control, c)
            control_residual = zero_layer(x_control)
            controlnet_outputs.append(control_residual)
        return controlnet_outputs

    def _backbone_forward(self, controlnet_outputs: list[Tensor], x: Tensor, c: Tensor) -> Tensor:
        """
        A forward pass for the frozen backbone, with ControlNet activations.

        :param controlnet_outputs: The activations of the ControlNet.
        :param x: The current latent vector (unconditioned).
        :param c: The conditioning tensor.
        :returns: The output of the frozen backbone.
        """
        for block in self.base_model.blocks:
            controlnet_out: Union[Tensor, float] = (
                controlnet_outputs.pop(0) if controlnet_outputs else 0.0
            )
            x = block(x + controlnet_out, c)  # (N, T, D)
        return x
