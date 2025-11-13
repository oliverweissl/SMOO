import gc
from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler, UNet2DModel, VQModel

from ._internal.models.autoencoder import vae_models
from ._internal.models.sit import SiT, SiT_models
from ._internal.utils import load_encoders


@dataclass
class LoadedSiTContainer:
    """A container for better return typing."""

    model: SiT
    vae: torch.nn.Module
    latents_scale: torch.Tensor
    latents_bias: torch.Tensor
    latent_size: int
    in_channels: int


def load_sit(
    image_resolution: int,
    num_classes: int,
    model_file: str,
    vae: str,
    model: str,
    encoder: str,
    device: torch.device,
) -> LoadedSiTContainer:
    """
    Load a SiT Diffusion model.

    :param image_resolution: Image resolution for generation.
    :param num_classes: Number of classes available in generation.
    :param model_file: Model file to load weights from.
    :param vae: The type of VAE model to use.
    :param model: The type of model to use.
    :param encoder: The type of encoder to use.
    :param device: The device to use.
    :return: The loaded diffusion model and more.
    :raises NotImplementedError: If VAE type is not supported.
    """
    state_dict = torch.load(model_file, weights_only=False)

    """Prepare VAE model."""
    if vae == "f8d4":
        latent_size = image_resolution // 8
        in_channels = 4
    elif vae == "f16d32":
        latent_size = image_resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError(f"VAE of type {vae} is not supported")

    vae_m: torch.nn.Module = vae_models[vae]()
    vae_m.load_state_dict(state_dict["vae"])
    vae_m.eval()
    vae_m.to(device)

    """Prepare SiT model with optimized loading."""
    encoders, _, _ = load_encoders(encoder, "cpu", image_resolution)
    z_dims = [encoder.embed_dim for encoder in encoders] if encoder != "None" else [0]

    # Immediate cleanup
    del encoders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_m = SiT_models[model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=0.1,
        z_dims=z_dims,
        encoder_depth=8,
        bn_momentum=0.1,
        fused_attn=True,
        qk_norm=False,
    )
    model_m.load_state_dict(state_dict["ema"])
    model_m.eval()
    model_m.to(device)

    latents_scale = (
        state_dict["ema"]["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
    )
    latents_bias = state_dict["ema"]["bn.running_mean"].view(1, in_channels, 1, 1).to(device)

    """Clean up with optimized memory management."""
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return_container = LoadedSiTContainer(
        model_m, vae_m, latents_scale, latents_bias, latent_size, in_channels
    )
    return return_container


def load_default_sit(model_file: str, device: torch.device) -> LoadedSiTContainer:
    """
    Loads the default SiT used for HyNeA.

    :param model_file: Model file to load weights from.
    :param device: CUDA device to use if available.
    :return: Loaded diffusion model and more.
    """
    container = load_sit(
        model_file=model_file,
        image_resolution=256,
        num_classes=1000,
        vae="f16d32",
        model="SiT-XL/1",
        encoder="dinov2-vit-b",
        device=device,
    )
    return container


def load_ldm_celebhq(
    device: torch.device,
) -> tuple[UNet2DModel, VQModel, DDIMScheduler]:
    """
    Loads a pretrained CELEB-HQ LDM (Unet) model.

    :param device: CUDA device to use.
    :returns: The U-Net, VQ-VAE and DDIM scheduler.
    """
    """This is an old model -> No safetensors available on HF."""
    unet = UNet2DModel.from_pretrained(
        "CompVis/ldm-celebahq-256", subfolder="unet", use_safetensors=False
    )
    vqvae = VQModel.from_pretrained(
        "CompVis/ldm-celebahq-256", subfolder="vqvae", use_safetensors=False
    )
    scheduler = DDIMScheduler.from_pretrained(
        "CompVis/ldm-celebahq-256", subfolder="scheduler", use_safetensors=False
    )

    unet.to(device=device)
    vqvae.to(device=device)

    unet.eval()
    vqvae.eval()
    return unet, vqvae, scheduler
