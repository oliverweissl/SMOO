from copy import deepcopy

import torch
from diffusers import StableDiffusionControlNetPipeline
from torch import Tensor, nn

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
        control: Tensor,
        prompts: list[str],
        timesteps: int,
        x: Tensor,
    ) -> Tensor:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param control: The control signal to other control nets.
        :param prompts: Prompts to generate diffusion steps for.
        :param timesteps: Number of steps in the denoising.
        :param x: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and empty tensor as there are no classes here.
        """
        batch_size = len(prompts)
        controlnet = (
            self._pipe.controlnet._orig_mod
            if hasattr(self._pipe.controlnet, "_orig_mod")
            else self._pipe.controlnet
        )

        do_cfg, guess_mode = (
            self._pipe.do_classifier_free_guidance,
            controlnet.config.global_pool_conditions,
        )
        timesteps = timesteps or self._diffusion_steps
        control = self._pipe.prepare_image(
            control,
            None,
            None,
            guess_mode=guess_mode,
            do_classifier_free_guidance=do_cfg,
            device=self._device,
            dtype=controlnet.dtype,
            num_images_per_prompt=1,
            batch_size=batch_size,
        )

        y_cur, negative_prompt_embeds = self._pipe.encode_prompt(
            prompts,
            self._device,
            1,
            do_cfg,
            [self._negative_prompt] * batch_size,
        )

        if do_cfg:
            y_cur = torch.cat([negative_prompt_embeds, y_cur])

        x_cur = (
            x.to(self._device, controlnet.dtype)
            if x is not None
            else torch.randn(
                batch_size,
                self._pipe.unet.config["in_channels"],
                self._pipe.unet.sample_size,
                self._pipe.unet.sample_size,
                device=self._device,
                dtype=controlnet.dtype,
            )
            * self._pipe.scheduler.init_noise_sigma
        )

        self._pipe.scheduler.set_timesteps(num_inference_steps=timesteps)

        timestep_cond = None
        if self._pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self._pipe.guidance_scale - 1).repeat(batch_size)
            timestep_cond = self._pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self._pipe.unet.config.time_cond_proj_dim
            ).to(device=self._device, dtype=x_cur.dtype)

        for i, t in enumerate(self._pipe.scheduler.timesteps):
            if self._pipe.interrupt:
                continue

            latents = torch.cat([x_cur] * 2) if do_cfg else x_cur
            ldm_input = self._pipe.scheduler.scale_model_input(latents, t)

            if guess_mode and do_cfg:
                cn_input = self._pipe.scheduler.scale_model_input(latents, t)
                controlnet_prompt_embeds = y_cur.chunk(2)[1]
            else:
                cn_input = ldm_input
                controlnet_prompt_embeds = y_cur

            down_block_res_samples, mid_block_res_sample = controlnet(
                cn_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=control,
                conditioning_scale=1.0,
                guess_mode=guess_mode,
                return_dict=False,
            )

            if guess_mode and do_cfg:
                down_block_res_samples = [
                    torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
                ]
                mid_block_res_sample = torch.cat(
                    [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                )

            # predict the noise residual
            noise_pred = self._pipe.unet(
                ldm_input,
                t,
                encoder_hidden_states=y_cur,
                timestep_cond=timestep_cond,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self._pipe.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            x_cur, *_ = self._pipe.scheduler.step(noise_pred, t, x_cur, return_dict=False)

            del down_block_res_samples, mid_block_res_sample, noise_pred, ldm_input, cn_input
            torch.cuda.empty_cache()
        return x_cur
