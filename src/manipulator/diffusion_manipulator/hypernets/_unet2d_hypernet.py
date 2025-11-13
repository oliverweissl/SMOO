from copy import deepcopy
from typing import Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel
from torch import Tensor, nn

from ._hypernet import HyperNet
from .blocks import ControlProjector, ZeroConv2d


class UNet2DHyperNet(nn.Module, HyperNet):
    """A hypernet class for UNet2D models."""

    use_checkpoints: bool = True

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

        """Initialize the Hypernet stuff."""
        self.control_in = deepcopy(model.conv_in)
        self.control_down = deepcopy(model.down_blocks)
        self.control_mid = deepcopy(model.mid_block)

        for param in self._model.parameters():
            param.requires_grad_(False)  # Freeze parameters

        # Create zero-conv layers for all relevant layers.
        self.zero_in = ZeroConv2d(model.conv_in.out_channels)
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
            model.conv_in.in_channels,
            model.sample_size,
            model.sample_size,
        )
        self.control_projector = ControlProjector(
            input_shape=self.in_shape,
            control_shape=control_shape,
            device=self._model.device,
            dtype=self._model.dtype,
        )
        self.standardize_control = torch.nn.Tanh()

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
        timesteps: int = 50,
    ) -> Tensor:
        """
        Full denoising process - used for end-to-end training.

        :param x: (B, C, H, W) tensor of spatial inputs (latent representations of images or None).
        :param control: (B, *S) tensor of control tokens to use for the forward pass assumes range (-inf, inf).
        :param timesteps: (B, *S) tensor of timesteps to use for the forward pass.
        :returns: The results of the forward pass.
        """
        if x is None:
            x = torch.randn((control.size(0), *self.in_shape), device=control.device)

        control = self.standardize_control(control)
        self._scheduler.set_timesteps(timesteps)
        for t in self._scheduler.timesteps:
            residual = self._diffusion_step(x, control, t)
            x, *_ = self._scheduler.step(residual, t, x, eta=0.0, return_dict=False)
        return x

    def _diffusion_step(self, x: Tensor, control: Tensor, t: int | Tensor) -> Tensor:
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

        if not isinstance(t, Tensor):
            tt = torch.tensor([t], dtype=torch.long, device=x.device)
        else:
            tt = t[None].to(x.device) if len(t.shape) == 0 else t

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        tt = tt * torch.ones(x.shape[0], dtype=tt.dtype, device=tt.device)

        t_emb = self._model.time_proj(tt)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self._model.dtype)
        emb = self._model.time_embedding(t_emb)

        # Get control residuals and control x from down sampling control network.
        control_down_residuals, control_x = self._control_down(x, control, emb)
        # Get residuals and current x from down sampling network.
        down_residuals, x = self._down(x, emb)

        # 4. mid
        if self._model.mid_block is not None:
            x = self._eval_module(self._model.mid_block, x + control_x, emb)

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
            x = x + skip_sample

        if self._model.config.get("time_embedding_type") == "fourier":
            tt = tt.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            x = x / tt
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
        projected_control = self._eval_module(self.control_projector, control)
        x_conditioned = self._eval_module(self.control_in, x + projected_control)

        skip_sample = x
        x0 = self._eval_module(self.zero_in, x_conditioned)
        outputs_down = (x0,)

        for block, zeros in zip(self.control_down, self.zero_downs):
            # Caution! These need to be in order as they are parsed as args!
            # Key-words: hidden_states, temb, skip_sample.
            b_args = (
                (x_conditioned, emb, skip_sample)
                if hasattr(block, "skip_conv")
                else (x_conditioned, emb)
            )

            # This looks sketchy but is cool!
            # We unpack the functions outputs (can be 2 or 3), if there is only two we keep skip, sample the same.
            # If there is three outputs we will get 4 elements and as such we take the first 3 to update the variables.
            x_conditioned, res_samples, skip_sample = (
                *self._eval_module(block, *b_args, emb),
                skip_sample,
            )[:3]
            outputs_down += tuple([z(s) for z, s in zip(zeros, res_samples)])

        output_mid = self._eval_module(self.control_mid, x_conditioned, emb)
        output_mid = self._eval_module(self.zero_mid, output_mid)
        return outputs_down, output_mid

    def _down(self, x: Tensor, emb: Tensor) -> tuple[tuple[Tensor], Tensor]:
        """
        The forward pass through the down sampling network.

        :param x: The input.
        :param emb: The embedding (time).
        :returns: The residuals collected and the final x.
        """
        skip_sample = x
        x = self._eval_module(self._model.conv_in, x)

        down_block_res_samples = (x,)
        for block in self._model.down_blocks:
            # Caution! These need to be in order as they are parsed as args!
            # Key-words: hidden_states, temb, skip_sample.
            b_args = (x, emb, skip_sample) if hasattr(block, "skip_conv") else (x, emb)

            # This looks sketchy but is cool!
            # We unpack the functions outputs (can be 2 or 3), if there is only two we keep skip, sample the same.
            # If there is three outputs we will get 4 elements and as such we take the first 3 to update the variables.
            x, res_samples, skip_sample = (*self._eval_module(block, *b_args), skip_sample)[:3]
            down_block_res_samples += res_samples
        return down_block_res_samples, x
