from copy import deepcopy
from math import prod
from typing import Optional, Union

import torch
from diffusers import DDIMScheduler, UNet2DModel
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class ZeroConvBlock(nn.Conv2d):
    """Zero Convolution Block."""

    def __init__(self, channels: int) -> None:
        """
        Initialize a Zero Convolution Block.

        :param channels: Number of channels in the input.
        """
        super().__init__(channels, channels, 1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ControlProjector(nn.Module):
    """Control Projector."""

    def __init__(self, input_shape: tuple[int, ...], control_shape: tuple[int, ...]) -> None:
        """
        Initialize the Control Projector.

        :param input_shape: Shape of the input for the UNet2D (excluding batch_dim).
        :param control_shape: Shape of the control input (excluding batch_dim).
        """
        super().__init__()
        # Embed the control shape into correct dimensionality for the reshape later.
        self.embedder = nn.Linear(prod(control_shape), prod(input_shape), bias=False)
        self.input_shape = input_shape
        self.projector = ZeroConvBlock(input_shape[0])

    def forward(self, control: Tensor) -> Tensor:
        """
        Project the control input to correct dimensions.

        :param control: Control input.
        :return: Projected control input.
        """
        b = control.size(0)
        flat = control.view(b, -1)
        x = self.embedder(flat)
        x = x.view(b, *self.input_shape)
        return self.projector(x)


class UNet2DHyperNet(nn.Module):
    """A hypernet class for UNet2D models."""

    def __init__(
        self,
        model: UNet2DModel,
        scheduler: DDIMScheduler,
        control_shape: tuple[int, ...],
    ) -> None:
        """
        Initialize a Hypernet class for UNet2D models.

        :param model: UNet2D model.
        :param scheduler: The scheduler used for this model.
        :param control_shape: Shape of the control input (excluding batch_dim).
        """
        super().__init__()
        """Store models and scheduler + Freeze weights."""
        self._model = model
        self._scheduler = scheduler

        for param in self._model.parameters():
            param.requires_grad_(False)  # Freeze parameters

        """Initialize the Hypernet stuff."""
        self.control_in = deepcopy(model.conv_in)
        self.control_down = deepcopy(model.down_blocks)
        self.control_mid = deepcopy(model.mid_block)

        # Create zero-conv layers for all relevant layers.
        self.zero_in = ZeroConvBlock(model.conv_in.out_channels)
        zero_downs = []
        for down_block in self.control_down:
            module_list = []
            for resnet in down_block.resnets:
                module_list.append(ZeroConvBlock(resnet.conv2.out_channels))

            if down_block.downsamplers is not None:
                for downsampler in down_block.downsamplers:
                    module_list.append(ZeroConvBlock(downsampler.out_channels))

            zero_downs.append(nn.ModuleList(module_list))
        self.zero_downs = nn.ModuleList(zero_downs)

        self.zero_mid = ZeroConvBlock(self.control_mid.resnets[-1].conv2.out_channels)

        # The shape of the latent inputs to the LDM.
        self.in_shape: tuple[int, int, int] = (
            model.conv_in.in_channels,
            model.sample_size,
            model.sample_size,
        )
        self.control_projector = ControlProjector(
            input_shape=self.in_shape, control_shape=control_shape
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        """
        Parse all trainable parameters in the model.

        :returns: A list of trainable parameters in the model (Control-Layers, Zero-Layers, Control-Projector).
        """
        return [
            *self.control_in.parameters(),
            *self.control_down.parameters(),
            *self.control_mid.parameters(),
            *self.zero_in.parameters(),
            *self.zero_downs.parameters(),
            *self.zero_mid.parameters(),
            *self.control_projector.parameters(),
        ]

    def forward(
        self,
        control: Tensor,
        x: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full denoising process - used for end-to-end training.

        :param x: (B, C, H, W) tensor of spatial inputs (latent representations of images or None).
        :param control: (B, *S) tensor of control tokens to use for the forward pass.
        :returns: The results of the forward pass.
        """
        device = control.device
        x = x or torch.randn((control.size(0), *self.in_shape)).to(device)

        latents = x
        for t in self._scheduler.timesteps:
            residual = self._diffusion_step(latents, control, t)
            latents = self._scheduler.step(residual, t, latents, eta=0.0)["prev_sample"]
        return latents

    def _diffusion_step(self, x: Tensor, control: Tensor, t: Union[Tensor, float, int]) -> Tensor:
        """
        A single diffusion step including control.

        Based on diffusers.UNet2DModel.forward().

        :param x: The input.
        :param t: The time step.
        :param control: The control token.
        :returns: The results of the diffusion step.
        """
        # 0. center input if necessary
        if self._model.config.get("center_input_sample", False):
            x = 2 * x - 1.0

        # 1. time
        if not isinstance(t, Tensor):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        else:
            if len(t.shape) == 0:
                t = t[None].to(x.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)

        t_emb = self._model.time_proj(t)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self._model.dtype)
        emb = self._model.time_embedding(t_emb)

        # Get control residuals and control x from down sampling control network.
        control_down_residuals, control_x = checkpoint(
            self._control_down, x, control, emb, use_reentrant=False
        )
        # Get residuals and current x from down sampling network.
        down_residuals, x = self._down(x, emb)

        # 4. mid
        if self._model.mid_block is not None:
            x = self._model.mid_block(x + control_x, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self._model.up_blocks:
            res_samples = down_residuals[-len(upsample_block.resnets) :]
            res_control_samples = control_down_residuals[-len(upsample_block.resnets) :]

            down_residuals = down_residuals[: -len(upsample_block.resnets)]
            control_down_residuals = control_down_residuals[: -len(upsample_block.resnets)]

            # Here we add control signals to the residual connections.
            res_inputs = tuple([s + c for s, c in zip(res_samples, res_control_samples)])
            if hasattr(upsample_block, "skip_conv"):
                x, skip_sample = upsample_block(x, res_inputs, emb, skip_sample)
            else:
                x = upsample_block(x, res_inputs, emb)

        # 6. post-process
        x = self._model.conv_norm_out(x)
        x = self._model.conv_act(x)
        x = self._model.conv_out(x)

        if skip_sample is not None:
            x += skip_sample

        if self._model.config.get("time_embedding_type") == "fourier":
            t = t.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            x = x / t
        return x

    def _control_down(
        self, x: Tensor, control: Tensor, emb: Tensor
    ) -> tuple[tuple[Tensor], Tensor]:
        """
        A forward pass only in the down sampling control network.

        :param x: The input.
        :param control: The control token.
        :param emb: The embedding.
        :returns: The results of the forward pass.
        """
        projected_control = self.control_projector(control)
        x_conditioned = x + projected_control
        x_conditioned = self.control_in(x_conditioned)

        skip_sample = x
        outputs_down = (self.zero_in(x_conditioned),)

        for block, zeros in zip(self.control_down, self.zero_downs):
            if hasattr(block, "skip_conv"):
                x_conditioned, res_samples, skip_sample = block(
                    hidden_states=x_conditioned, temb=emb, skip_sample=skip_sample
                )
            else:
                x_conditioned, res_samples = block(hidden_states=x_conditioned, temb=emb)
            outputs_down += tuple([z(s) for z, s in zip(zeros.modules(), res_samples)])

        output_mid = self.control_mid(x_conditioned, emb)
        return outputs_down, self.zero_mid(output_mid)

    def _down(self, x: Tensor, emb: Tensor) -> tuple[tuple[Tensor], Tensor]:
        """
        The forward pass through the down sampling network.

        :param x: The input.
        :param emb: The embedding (time).
        :returns: The resisuals collected and the final x.
        """
        skip_sample = x
        x = self._model.conv_in(x)

        down_block_res_samples = (x,)
        for downsample_block in self._model.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                x, res_samples, skip_sample = downsample_block(
                    hidden_states=x, temb=emb, skip_sample=skip_sample
                )
            else:
                x, res_samples = downsample_block(hidden_states=x, temb=emb)
            down_block_res_samples += res_samples
        return down_block_res_samples, x

    def set_timesteps(self, steps: int) -> None:
        """
        Set timesteps for scheduler.

        :param steps: The number of timesteps.
        """
        self._scheduler.set_timesteps(steps)
