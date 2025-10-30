from copy import deepcopy
from typing import Optional

import torch
from diffusers import StableDiffusionControlNetPipeline
from torch import nn

from .blocks import ControlProjector, ZeroConv2d


class SDCNHyperNet(nn.Module):
    """A hypernet class for UNet2D models."""

    use_checkpoints: bool = True

    def __init__(
        self,
        pipe: StableDiffusionControlNetPipeline,
        control_shape: tuple[int, ...],
    ) -> None:
        """
        Initialize a Hypernet class for UNet2D models.

        :param pipe: The pipeline to make the hypernet for.
        :param control_shape: Shape of the control input (excluding batch_dim).
        """
        super().__init__()
        """Store models and scheduler + Freeze weights."""
        self._model = pipe.unet
        self._scheduler = pipe.scheduler

        """Initialize the Hypernet stuff."""
        self.control_in = deepcopy(self._model.conv_in)
        self.control_down = deepcopy(self._model.down_blocks)
        self.control_mid = deepcopy(self._model.mid_block)

        for param in self._model.parameters():
            param.requires_grad_(False)  # Freeze parameters

        # Create zero-conv layers for all relevant layers.
        self.zero_in = ZeroConv2d(self._model.conv_in.out_channels)
        zero_downs = []
        for down_block in self.control_down:
            module_list = []
            for resnet in down_block.resnets:
                module_list.append(ZeroConv2d(resnet.conv2.out_channels))

            if down_block.downsamplers is not None:
                for downsampler in down_block.downsamplers:
                    module_list.append(ZeroConv2d(downsampler.out_channels))

            zero_downs.append(nn.ModuleList(module_list))
        self.zero_downs = nn.ModuleList(zero_downs)

        self.zero_mid = ZeroConv2d(self.control_mid.resnets[-1].conv2.out_channels)

        # The shape of the latent inputs to the LDM.
        self.in_shape: tuple[int, int, int] = (
            self._model.conv_in.in_channels,
            self._model.sample_size,
            self._model.sample_size,
        )
        self.control_projector = ControlProjector(
            input_shape=self.in_shape, control_shape=control_shape
        )
        self.bound_control = torch.nn.Tanh()

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
        control: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        timesteps: int = 50,
    ) -> torch.Tensor:
        """
        Full denoising process for ControlNet - used for end-to-end training.

        :param x: (B, C, H, W) tensor of spatial inputs (latent representations of images or None).
        :param control: (B, *S) tensor of control tokens to use for the forward pass.
        :param timesteps: Number of diffusion timesteps to use.
        :returns: The results of the forward pass.
        """
        if x is None:
            x = torch.randn(
                (control.size(0), *self.in_shape), device=control.device, dtype=control.dtype
            )

        # Standardize control input
        control = self.bound_control(control)

        # Set up scheduler
        self._scheduler.set_timesteps(timesteps)

        for t in self._scheduler.timesteps:
            # Scale model input
            x_scaled = self._scheduler.scale_model_input(x, t)

            # Project control signal
            projected_control = self.control_projector(control)

            # Apply control conditioning through the control network
            x_conditioned = x_scaled + projected_control
            x_conditioned = self.control_in(x_conditioned)

            # Get control residuals from down blocks
            control_down_residuals = []
            skip_sample = x_scaled

            for block, zeros in zip(self.control_down, self.zero_downs):
                if hasattr(block, "skip_conv"):
                    x_conditioned, res_samples, skip_sample = block(x_conditioned, t, skip_sample)
                else:
                    x_conditioned, res_samples = block(x_conditioned, t)

                # Apply zero convolutions to residuals
                control_residuals = [z(s) for z, s in zip(zeros, res_samples)]
                control_down_residuals.extend(control_residuals)

            # Process through middle block
            control_mid_residual = self.zero_mid(self.control_mid(x_conditioned, t))

            # Run the main UNet with control residuals
            noise_pred = self._model(
                x_scaled,
                t,
                down_block_additional_residuals=control_down_residuals,
                mid_block_additional_residual=control_mid_residual,
                return_dict=False,
            )[0]

            # Step the scheduler
            x = self._scheduler.step(noise_pred, t, x, return_dict=False)[0]

        return x
