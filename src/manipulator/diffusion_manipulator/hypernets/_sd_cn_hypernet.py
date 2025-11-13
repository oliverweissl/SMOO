from typing import Optional

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from torch import Tensor, nn

from ._hypernet import HyperNet
from .blocks import ControlProjector


class SDCNHyperNet(nn.Module, HyperNet):
    """A hypernet class for UNet2D models."""

    use_checkpoints: bool = True  # Default true as these are big models.

    def __init__(
        self,
        pipe: StableDiffusionControlNetPipeline,
        control_shape: tuple[int, ...],
        negative_prompts: str | list[str],
    ) -> None:
        """
        Initialize a Hypernet class for UNet2D models.

        :param pipe: The pipeline to make the hypernet for.
        :param control_shape: Shape of the control input (excluding batch_dim).
        :param negative_prompts: The negative prompts to use.
        """
        super().__init__()
        self._negative_prompts = negative_prompts

        """Store pipeline parameters."""
        self._device = pipe.device
        self._dtype = pipe.unet.dtype
        self._do_cfg = pipe.do_classifier_free_guidance
        self._guidance_scale = pipe.guidance_scale

        """Get useful pipeline functions."""
        self._embed_guidance = pipe.get_guidance_scale_embedding
        self._prepare_image = pipe.prepare_image
        self._encode_prompt = pipe.encode_prompt

        """Store models and scheduler + Freeze weights."""
        self._controlnet = (
            pipe.controlnet._orig_mod if hasattr(pipe.controlnet, "_orig_mod") else pipe.controlnet
        )
        self._model = pipe.unet
        self._scheduler = pipe.scheduler

        for param in self._model.parameters():
            param.requires_grad_(False)  # Freeze parameters
        for param in self._controlnet.parameters():
            param.requires_grad_(False)  # Freeze parameters

        """Initialize the Hypernet stuff -> super easy here with UNet Architectures."""
        hynea_controlnet = ControlNetModel.from_unet(self._model)
        self.control_down = hynea_controlnet.down_blocks
        self.control_mid = hynea_controlnet.mid_block

        # Create zero-conv layers for all relevant layers
        self.zero_downs = hynea_controlnet.controlnet_down_blocks
        self.zero_mid = hynea_controlnet.controlnet_mid_block

        del hynea_controlnet
        torch.cuda.empty_cache()

        # The shape of the latent inputs to the LDM.
        self.in_shape: tuple[int, int, int] = (
            self._model.conv_in.out_channels,
            self._model.sample_size,
            self._model.sample_size,
        )

        self.control_projector = ControlProjector(
            input_shape=self.in_shape,
            control_shape=control_shape,
            dtype=self._dtype,
            device=self._device,
        )

        """Cast self and all submodules to right dtype + device."""
        self.to(device=self._device, dtype=self._dtype)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """
        Parse all trainable parameters in the model.

        :returns: A list of trainable parameters in the model (Control-Layers, Zero-Layers, Control-Projector).
        """
        return [
            *self.control_down.parameters(),
            *self.control_mid.parameters(),
            *self.zero_downs.parameters(),
            *self.zero_mid.parameters(),
            *self.control_projector.parameters(),
        ]

    def forward(
        self,
        control: Tensor,
        control_signal: Tensor,
        prompts: list[str],
        timesteps: int,
        x: Tensor,
    ) -> Tensor:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param control: The control input for the HyNeA network.
        :param control_signal: The control signal to other ControlNets.
        :param prompts: Prompts to generate diffusion steps for.
        :param timesteps: Number of steps in the denoising.
        :param x: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and empty tensor as there are no classes here.
        """
        """Ensure inputs are on the same device and dtype as the network."""
        control = control.to(device=self.device, dtype=self.dtype)
        control_signal = control_signal.to(device=self.device, dtype=self.dtype)
        x = x.to(device=self.device, dtype=self.dtype)

        batch_size = len(prompts)

        guess_mode = self._controlnet.config.global_pool_conditions
        timesteps = timesteps or self._diffusion_steps
        control_signal = self._prepare_image(
            control_signal,
            None,
            None,
            guess_mode=guess_mode,
            do_classifier_free_guidance=self._do_cfg,
            device=self.device,
            dtype=self.dtype,
            num_images_per_prompt=1,
            batch_size=batch_size,
        )

        y_cur, negative_prompt_embeds = self._encode_prompt(
            prompts,
            self.device,
            1,
            self._do_cfg,
            (
                [self._negative_prompts] * batch_size
                if isinstance(self._negative_prompts, str)
                else self._negative_prompts
            ),
        )

        if self._do_cfg:
            y_cur = torch.cat([negative_prompt_embeds, y_cur])

        x_cur = (
            x.to(self.device, self.dtype)
            if x is not None
            else torch.randn(
                batch_size,
                self._model.config["in_channels"],
                self._model.sample_size,
                self._model.sample_size,
                device=self.device,
                dtype=self.dtype,
            )
            * self._scheduler.init_noise_sigma
        )

        self._scheduler.set_timesteps(num_inference_steps=timesteps)

        timestep_cond = None
        if self._model.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self._guidance_scale - 1).repeat(batch_size)
            timestep_cond = self._embed_guidance(
                guidance_scale_tensor, embedding_dim=self._model.config.time_cond_proj_dim
            ).to(device=self.device, dtype=x_cur.dtype)

        for i, t in enumerate(self._scheduler.timesteps):
            latents = torch.cat([x_cur] * 2) if self._do_cfg else x_cur
            ldm_input = self._scheduler.scale_model_input(latents, t)

            """Evaluate the ControlNet Networks."""
            controlnet_mid, controlnet_down = self._controlnet_forward(
                latents, ldm_input, y_cur, t, control_signal
            )

            """Evaluate the HyNeA Networks."""
            hynea_mid, hynea_down = self._hynea_forward(ldm_input, y_cur, t, control)

            """Add HyNeA control to the ControlNet control signal."""
            mid_res = controlnet_mid + hynea_mid

            down_res = [
                sample + hynea_sample for sample, hynea_sample in zip(controlnet_down, hynea_down)
            ]

            """Add additional Dimensions for CFG."""
            if guess_mode and self._do_cfg:
                down_res = [torch.cat([torch.zeros_like(d), d]) for d in down_res]
                mid_res = torch.cat([torch.zeros_like(mid_res), mid_res])

            noise_pred = self._unet_forward(ldm_input, y_cur, t, timestep_cond, down_res, mid_res)

            if self._do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self._guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            x_cur, *_ = self._scheduler.step(noise_pred, t, x_cur, return_dict=False)

            del down_res, mid_res, noise_pred, ldm_input
            torch.cuda.empty_cache()
        return x_cur

    def _controlnet_forward(
        self, z: Tensor, x: Tensor, y: Tensor, t: Tensor, control_signal: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        """
        The forward pass of the ControlNet Network.

        :param z: The latent vector.
        :param x: The input to the model.
        :param y: The embedded prompt.
        :param t: The raw time step.
        :param control_signal: The control signal to the ControlNet.
        :return: The residual output of the mid-block and the residual outputs of the down-blocks.
        """
        guess_mode = self._controlnet.config.global_pool_conditions
        if guess_mode and self._do_cfg:
            cn_input = self._scheduler.scale_model_input(z, t)
            controlnet_prompt_embeds = y.chunk(2)[1]
        else:
            cn_input = x
            controlnet_prompt_embeds = y

        down_block_res_samples, mid_block_res_sample = self._eval_module(
            self._controlnet,
            cn_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_signal,
            conditioning_scale=1.0,
            guess_mode=guess_mode,
            return_dict=False,
        )
        return mid_block_res_sample, down_block_res_samples

    def _hynea_forward(
        self, x: Tensor, y: Tensor, t: Tensor, control: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        """
        The forward pass through the HyNeA Hypernet.
        Based on ControlNetModel from diffusers.

        :param x: The input to the HyNeA Hypernet.
        :param y: The embeddings of the prompt.
        :param t: The raw timestep for denoising.
        :param control: The control input to the HyNeA Hypernet.
        :returns: A tuple containing the residual output of the Mid-Block and the residual outputs of the Down-Blocks.
        """
        hynea_kwargs = {"encoder_hidden_states": y}

        timesteps = t
        timesteps = timesteps.expand(x.shape[0])
        t_emb = self._controlnet.time_proj(timesteps).to(device=x.device, dtype=x.dtype)
        emb = self._controlnet.time_embedding(t_emb)

        x = self._model.conv_in(x)
        co = self.control_projector(control)
        x_control = x + co

        """Get the residual outputs of the HyNeA Hypernet."""
        hynea_down = [x_control]

        for block in self.control_down:
            block_kwargs = (
                hynea_kwargs
                if hasattr(block, "has_cross_attention") and block.has_cross_attention
                else {}
            )
            x_control, x_res = self._eval_module(block, x_control, emb, **block_kwargs)
            hynea_down.extend(x_res)

        mid_kwargs = (
            hynea_kwargs
            if hasattr(self.control_mid, "has_cross_attention")
            and self.control_mid.has_cross_attention
            else {}
        )
        x_control = self._eval_module(self.control_mid, x_control, emb, **mid_kwargs)

        """Modulate HyNeA Hypernet residuals with ZeroBlocks."""
        hynea_down_zerod = []
        for block_res_samples, zero_block in zip(hynea_down, self.zero_downs):
            hynea_down_zerod.append(zero_block(block_res_samples))
        hynea_down = hynea_down_zerod

        x_control = self.zero_mid(x_control)

        if self._controlnet.config.global_pool_conditions:
            hynea_down = [torch.mean(sample, dim=(2, 3), keepdim=True) for sample in hynea_down]
            x_control = torch.mean(x_control, dim=(2, 3), keepdim=True)
        return x_control, hynea_down

    def _unet_forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        t_cond: Optional[Tensor],
        res_down: list[Tensor],
        res_mid: Tensor,
    ) -> Tensor:
        """
        Forward pass for the UNet model.

        :param x: The current input to the model (noise).
        :param y: The encoded prompt to the model.
        :param t: The current raw time step (not encoded).
        :param t_cond: A condition for the current time step (unclear).
        :param res_down: A list of residual outputs of down-blocks from the Hypernets.
        :param res_mid: A residual output of the mid-block from the Hypernets.
        :return: The predicted noise from the model.
        """
        noise_pred, *_ = self._eval_module(
            self._model,
            x,
            t,
            encoder_hidden_states=y,
            timestep_cond=t_cond,
            down_block_additional_residuals=res_down,
            mid_block_additional_residual=res_mid,
            return_dict=False,
        )
        return noise_pred
