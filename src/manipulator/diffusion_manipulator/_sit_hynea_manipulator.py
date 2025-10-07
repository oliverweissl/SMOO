import gc
from typing import Optional, Union

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn

from ._sit_control_net import SiTControlNet
from ._sit_manipulator import SiTManipulator


class SitHyNeAManipulator(SiTManipulator):  # TODO: probably should not depend on SiT Manipulator!
    """A trainer class for the ControlNet."""

    _control_net: SiTControlNet

    def __init__(
        self,
        model_file: str,
        control_shape: tuple[int, ...],
        cfg_scale: float = 1.5,
        batch_size: int = 0,
        device: Union[torch.device, None] = None,
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
        super().__init__(
            model_file=model_file,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            device=device,
            require_grad=True,
        )
        self._control_shape = control_shape
        self.make_fresh_control_net()

        self.noise_loss = nn.MSELoss()
        self.scheduler = DDPMScheduler(
            num_train_timesteps=50, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"
        )
        self._n_steps = diffusion_steps

    def make_fresh_control_net(self) -> None:
        """Create a new ControlNet for the current model. ATTENTION: Deletes old one if exists!."""
        if hasattr(self, "_control_net"):
            del self._control_net
            gc.collect()
            torch.cuda.empty_cache()
        self._control_net = SiTControlNet(self._model, self._control_shape)
        self._control_net.to(self._device)

    def _sample_control(
        self,
        t: Tensor,
        x: Tensor,
        y: Tensor,
        control: Tensor,
        step: float,
        guidance_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> Tensor:
        """
        Sampling new outputs for controlnet.

        :param t: The current time step.
        :param x: The current state of the diffusion process.
        :param y: The current class embedding of the diffusion process.
        :param control: The control map.
        :param step: The step size.
        :param guidance_bounds: Guidance bounds for conditions in the sampling.
        :returns: The sampled outputs for the current timestep.
        """
        cond = self._cfg > 1.0 and guidance_bounds[1] >= t >= guidance_bounds[0]

        t_curr = torch.full(size=(y.size(0),), fill_value=t.item(), device=self._device)
        if cond:
            model_input = x.repeat(2, *([1] * (x.ndim - 1)))
            null_embedding_cache = self._embed_y([1000] * y.shape[0])
            y_curr = torch.cat((y, null_embedding_cache), dim=0)
            t_curr = t_curr.repeat(2, *([1] * (t_curr.ndim - 1)))
            control_curr = control.repeat(2, *([1] * (control.ndim - 1)))
        else:
            model_input, y_curr, control_curr = x, y, control

        d_cur = self._control_net.inference(x=model_input, t=t_curr, y=y_curr, control=control_curr)
        if cond:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + self._cfg * (d_cur_cond - d_cur_uncond)
        return x + step * d_cur

    def get_diff_steps_control(
        self,
        class_labels: list[int],
        n_steps: int = 50,
        x_0: Optional[Tensor] = None,
        control: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param class_labels: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising.
        :param x_0: Optional starting latent vector if sampled differently.
        :param control: An optional control tensor for genreation of steps.
        :returns: A list of latent vectors through denoising and the class embedding.
        """
        batch_size = len(class_labels)

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

        control = (
            control.to(self._device)
            if control is not None
            else torch.randn(batch_size, *self._control_shape, device=self._device)
        )

        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)
        y_cur = self._embed_y(class_labels)

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
            x_cur = self._sample_control(
                t=t_cur, x=x_cur, y=y_cur, control=control, step=t_next - t_cur
            )
            xs[i + 1] = x_cur

        return xs.detach(), y_cur

    def manipulate(self, x: Tensor, y: list[int], c: Tensor) -> Tensor:
        y_cur = self._embed_y(y)
        y_null = self._embed_y([1000] * y_cur.shape[0])
        x = self._control_net.forward_full(
            x=x, y=y_cur, control=c, cfg=self._cfg, guidance_bounds=(0.0, 1.0), y_null=y_null
        )
        return x

    @property
    def control_net(self) -> SiTControlNet:
        """
        Get the controlnet used.

        :return: The controlnet used.
        """
        return self._control_net

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing.

        :param enable: Whether to enable gradient checkpointing.
        """
        if enable:
            self._control_net.gradient_checkpointing_enable()
        else:
            self._control_net.gradient_checkpointing_disable()
