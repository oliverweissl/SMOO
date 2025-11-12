import gc
import logging
from typing import Optional

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from torch import Tensor

from . import DiffusionCandidateList
from ._diffusion_manipulator import DiffusionManipulator
from ._utils import prepare_cuda
from .hypernets import SDCNHyperNet


class SDCNHyNeAManipulator(DiffusionManipulator):
    """A Manipulator class designed for diffusors-library ControlNet Pipelines."""

    _device: torch.device

    """Models used."""
    _pipe: StableDiffusionControlNetPipeline
    _hyper_net: SDCNHyperNet

    def __init__(
        self,
        pipeline_path: str,
        control_shape: tuple[int, ...],
        batch_size: int = 0,
        diffusion_steps: int = 50,
        device: Optional[torch.device] = None,
        controlnet_path: Optional[str] = None,
        guidance_scale: float = 9.0,
    ) -> None:
        """
        Initialize the LDM ControlNet Manipulator.

        :param pipeline_path: Path to the pipeline to use.
        :param control_shape: Shape of the control signal.
        :param batch_size: Batch size (0 means all - Default).
        :param diffusion_steps: Diffusion steps to take in denoising.
        :param device: Device to use for compute.
        :param controlnet_path: Path to the controlnet if specified seperately.
        :param guidance_scale: Guidance scale factor.
        """
        self.control_shape = control_shape
        self._device = prepare_cuda(device, True)
        self._batch_size = batch_size
        self._diffusion_steps = diffusion_steps

        controlnet = (
            ControlNetModel.from_pretrained(controlnet_path)
            if controlnet_path is not None
            else None
        )
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pipeline_path, controlnet=controlnet
        )

        self._pipe.to(self._device)
        self._pipe.enable_xformers_memory_efficient_attention()

        self._pipe._guidance_scale = guidance_scale
        self._pipe._interrupt = False

        for p in self._pipe.vae.parameters():
            p.requires_grad_(False)  # Freeze parameters

        self._negative_prompt = "blurry, distorted, ugly, low quality, cartoon, sketch"
        self.make_fresh_hyper_net()

    def make_fresh_hyper_net(self) -> None:
        """Create a new ControlNet for the current model. ATTENTION: Deletes old one if exists!."""
        if hasattr(self, "_hyper_net"):
            del self._hyper_net
            gc.collect()
            torch.cuda.empty_cache()
        self._hyper_net = SDCNHyperNet(
            pipe=self._pipe,
            control_shape=self.control_shape,
            negative_prompts=self._negative_prompt,
        )
        self._hyper_net.to(self._device)

    def manipulate(self, candidates: DiffusionCandidateList, **kwargs) -> Tensor:
        """
        Manipulate inputs with their respective control signals.

        :param candidates: The candidates to manipulate.
        :param kwargs: Additional KW-Args, use `timesteps: int` to modify default 50 diffusion steps.
        :return: The sampled outputs.
        """
        xs = []
        for c in candidates:
            assert (
                c.prompt is not None
            ), f"Error: prompt needed in candidate for {self.__class__.__name__}"
            assert (
                c.control is not None
            ), f"Error: control needed in candidate for {self.__class__.__name__}"
            assert (
                c.control_signal is not None
            ), f"Error: control_signal needed in candidate for {self.__class__.__name__}"

            xt = c.xt[0].unsqueeze(0)
            x = self._hyper_net.forward(
                x=xt,
                control=c.control,
                control_signal=c.control_signal,
                timesteps=self._diffusion_steps,
                prompts=c.prompt,
            )
            xs.append(x)
        return torch.cat(xs, dim=0)

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing.

        :param enable: Whether to enable gradient checkpointing.
        """
        self._hyper_net.use_checkpoints = enable

    def get_diff_steps(
        self,
        diff_input: tuple[Tensor, list[str]],
        n_steps: Optional[int] = None,
        x_0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param diff_input: The control signal to other control nets and the prompts.
        :param n_steps: Number of steps in the denoising.
        :param x_0: Optional starting latent vector if sampled differently.
        :returns: A list of latent vectors through denoising and empty tensor as there are no classes here.
        """
        control, prompts = diff_input

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
        n_steps = n_steps or self._diffusion_steps
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
            x_0.to(self._device, controlnet.dtype)
            if x_0 is not None
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
        xs_list = [x_cur.detach().cpu()]

        self._pipe.scheduler.set_timesteps(num_inference_steps=n_steps)

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

            # Use torch.no_grad() for inference to save memory
            with torch.no_grad():
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
                xs_list.append(x_cur.detach().cpu())

                del down_block_res_samples, mid_block_res_sample, noise_pred, ldm_input, cn_input
                torch.cuda.empty_cache()

        xs = torch.stack(xs_list)
        return xs, y_cur.detach()

    def get_images(self, z: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :param eps: The epsilon value to avoid gradient instabilities.
        :return: The decoded image, color-range [0,1].
        """
        z = z.to(self._device, dtype=self._pipe.vae.dtype)

        if not torch.isfinite(z).all():
            nan_count = torch.isnan(z).sum().item()
            inf_count = torch.isinf(z).sum().item()
            logging.warning(
                f"Latent z contains NaNs ({nan_count}) or Infs ({inf_count}), casting to valid range."
            )
            z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)

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
            image, *_ = self._pipe.vae.decode(
                z_chunk / self._pipe.vae.config.scaling_factor, return_dict=False
            )
            if not torch.isfinite(image).all():
                nan_count = torch.isnan(image).sum().item()
                inf_count = torch.isinf(image).sum().item()
                logging.warning(
                    f"VAE-Sampled Image contains NaNs ({nan_count}) or Infs ({inf_count}), casting to valid range."
                )
                image = torch.nan_to_num(image, nan=0.0, posinf=1, neginf=-1)

            image = (image * 0.5 + 0.5).clamp(0.0 + eps, 1.0 - eps)
            decoded.append(image)
        return torch.cat(decoded, dim=0)

    @property
    def hyper_net(self) -> SDCNHyperNet:
        """
        Get the HyperNet used.

        :return: The HyperNet used.
        """
        return self._hyper_net
