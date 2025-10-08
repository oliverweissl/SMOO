from typing import Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel
from torch import nn

from .. import Manipulator
from ._load_models import load_ldm_celebhq
from ._utils import prepare_cuda


class LDMHyNeAManipulator(Manipulator):
    """A trainer class for the LDM ControlNet."""

    _device: torch.device

    """Models used."""
    _vae: nn.Module
    _model: nn.Module
    _scheduler: DDIMScheduler
    _control_net: UNet2DModel

    def __init__(
        self,
        control_shape: tuple[int, ...],
        batch_size: int = 0,
        device: Optional[torch.device] = None,
        diffusion_steps: int = 50,
    ) -> None:
        """
        Initialize the LDM ControlNet Manipulator.

        :param control_shape: Shape of the control signal.
        :param batch_size: Batch size (0 means all - Default).
        :param device: Device to use for compute.
        :param diffusion_steps: Number of diffusion steps in scheduler.
        """
        self._device = prepare_cuda(device, True)
        self._model, self._vae, self._scheduler = load_ldm_celebhq(device=self._device)
        self._scheduler.set_batch_size(diffusion_steps)
