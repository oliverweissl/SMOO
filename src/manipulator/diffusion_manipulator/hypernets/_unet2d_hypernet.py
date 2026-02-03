from copy import deepcopy
from typing import Any, Optional

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
        guidance_scale: float = 1.0,
    ) -> None:
        """
        Initialize a Hypernet class for UNet2D models.

        :param model: UNet2D model.
        :param scheduler: The scheduler used for this model.
        :param control_shape: Shape of the control input (excluding batch_dim).
        :param guidance_scale: Classifier-free guidance scale (1.0=no CFG, >1.0 enables CFG).
        """
        super().__init__()
        """Store models and scheduler + Freeze weights."""
        self._model = model
        self._guidance_scale = guidance_scale
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
        encoder_hidden_states: Optional[Tensor] = None,
        start_step: int = 0,
        added_cond_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tensor:
        """
        Full denoising process - used for end-to-end training.

        :param x: (B, C, H, W) tensor of spatial inputs (latent representations of images or None).
        :param control: (B, *S) tensor of control tokens to use for the forward pass assumes range (-inf, inf).
        :param timesteps: Total number of timesteps in the schedule.
        :param encoder_hidden_states: Optional hidden states to include prompt conditioning.
        :param start_step: Which timestep index to start denoising from (0 = denoise all steps).
        :param added_cond_kwargs: Optional added conditioning kwargs for SDXL (text_embeds, time_ids).
        :returns: The results of the forward pass.
        """
        if x is None:
            x = torch.randn((control.size(0), *self.in_shape), device=control.device)

        control = self.standardize_control(control)
        self._scheduler.set_timesteps(timesteps)

        timesteps_to_denoise = self._scheduler.timesteps[start_step:]

        # Determine if CFG should be applied
        do_cfg = self._guidance_scale > 1.0 and (
            encoder_hidden_states is not None or added_cond_kwargs is not None
        )

        if do_cfg:
            # Duplicate inputs for conditional and unconditional passes
            x = torch.cat([x, x], dim=0)
            control = torch.cat([control, control], dim=0)

            # Create unconditional embeddings
            if encoder_hidden_states is not None:
                uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                encoder_hidden_states = torch.cat(
                    [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
                )

            if added_cond_kwargs is not None:
                # For SDXL: zero text_embeds, keep time_ids
                uncond_added_cond_kwargs = {
                    "text_embeds": torch.zeros_like(added_cond_kwargs["text_embeds"]),
                    "time_ids": added_cond_kwargs["time_ids"],
                }
                added_cond_kwargs = {
                    "text_embeds": torch.cat(
                        [uncond_added_cond_kwargs["text_embeds"], added_cond_kwargs["text_embeds"]],
                        dim=0,
                    ),
                    "time_ids": torch.cat(
                        [uncond_added_cond_kwargs["time_ids"], added_cond_kwargs["time_ids"]], dim=0
                    ),
                }

        # Denoising loop with CFG
        for t in timesteps_to_denoise:
            residual = self._diffusion_step(x, control, t, encoder_hidden_states, added_cond_kwargs)

            if do_cfg:
                # Split conditional and unconditional predictions
                residual_uncond, residual_cond = residual.chunk(2, dim=0)
                # Apply CFG formula
                residual = residual_uncond + self._guidance_scale * (
                    residual_cond - residual_uncond
                )
                # Only step with conditional latent
                x_cond = x.chunk(2, dim=0)[1]
                x_stepped, *_ = self._scheduler.step(
                    residual, t, x_cond, eta=0.0, return_dict=False
                )
                # Update both batches for next iteration
                x = torch.cat([x_stepped, x_stepped], dim=0)
            else:
                x, *_ = self._scheduler.step(residual, t, x, eta=0.0, return_dict=False)

        # Return only conditional result if CFG was used
        return x.chunk(2, dim=0)[1] if do_cfg else x

    def _diffusion_step(
        self,
        x: Tensor,
        control: Tensor,
        t: int | Tensor,
        encoder_hidden_states: Optional[Tensor],
        added_cond_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tensor:
        """
        A single diffusion step including control.

        Based on diffusers.UNet2DModel.forward().

        :param x: The input.
        :param t: The time step.
        :param control: The control token.
        :param encoder_hidden_states: Optional hidden states to include prompt conditioning.
        :param added_cond_kwargs: Optional added conditioning kwargs for SDXL.
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

        t_emb = t_emb.to(dtype=self._model.dtype)
        emb = self._model.time_embedding(t_emb)

        # Handle SDXL's additional embeddings
        if added_cond_kwargs is not None:
            aug_emb = self._model.get_aug_embed(
                emb=emb,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            )
            emb = emb + aug_emb

        control_down_residuals, control_x = self._control_down(
            x, control, emb, encoder_hidden_states=encoder_hidden_states
        )
        down_residuals, x = self._down(x, emb, encoder_hidden_states=encoder_hidden_states)

        # 4. mid
        if self._model.mid_block is not None:
            x = self._eval_module(
                self._model.mid_block,
                x + control_x,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )

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
                x, skip_sample = upsample_block(
                    x, res_inputs, emb, skip_sample, encoder_hidden_states=encoder_hidden_states
                )
            else:
                x = upsample_block(x, res_inputs, emb, encoder_hidden_states=encoder_hidden_states)

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
        self,
        x: Tensor,
        control: Tensor,
        emb: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
    ) -> tuple[tuple[Tensor], Tensor]:
        """
        A forward pass only in the down sampling control network.

        :param x: The input.
        :param control: The control token.
        :param emb: The embedding.
        :param encoder_hidden_states: Optional hidden states for cross-attention (text conditioning).
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
                *self._eval_module(block, *b_args, encoder_hidden_states=encoder_hidden_states),
                skip_sample,
            )[:3]
            outputs_down += tuple([z(s) for z, s in zip(zeros, res_samples)])

        output_mid = self._eval_module(
            self.control_mid, x_conditioned, emb, encoder_hidden_states=encoder_hidden_states
        )
        output_mid = self._eval_module(self.zero_mid, output_mid)
        return outputs_down, output_mid

    def _down(
        self, x: Tensor, emb: Tensor, encoder_hidden_states: Optional[Tensor]
    ) -> tuple[tuple[Tensor], Tensor]:
        """
        The forward pass through the down sampling network.

        :param x: The input.
        :param emb: The embedding (time).
        :param encoder_hidden_states: Additional hidden states for cross-attention (text conditioning).
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
            x, res_samples, skip_sample = (
                *self._eval_module(block, *b_args, encoder_hidden_states=encoder_hidden_states),
                skip_sample,
            )[:3]
            down_block_res_samples += res_samples
        return down_block_res_samples, x
