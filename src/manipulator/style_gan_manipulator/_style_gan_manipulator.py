from typing import Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .._manipulator import Manipulator
from ._internal import dnnlib
from ._internal.torch_utils.ops import upfirdn2d
from ._load_stylegan import load_stylegan
from ._mix_candidate import MixCandidateList


class StyleGANManipulator(Manipulator):
    """
    A class geared towards style-mixing using style layers in a StlyeGAN.

    This class is heavily influenced by the Renderer class in the StyleGAN3 repo.
    """

    _generator: torch.nn.Module
    _device: torch.device
    _has_input_transform: bool
    _noise_mode: str
    _interpolate: bool

    _mix_dims: Tensor  # Range for dimensions to mix styles with.

    """Some default parameters for the generation of w."""
    trunc_psi: float = 1.0
    trunc_cutoff: int = 0

    def __init__(
        self,
        generator: Union[torch.nn.Module, str],
        device: torch.device,
        mix_dims: tuple[int, int],
        interpolate: bool = True,
        noise_mode: str = "random",
        conditional: bool = True,
        batch_size: int = 0,
    ) -> None:
        """
        Initialize the StyleMixer object.

        :param generator: The generator network to use or the path to its pickle file.
        :param device: The torch device to use (should be cuda).
        :param mix_dims: The w-dimensions to use for mixing (range index).
        :param noise_mode: The noise mode to be used for generation (const, random).
        :param interpolate: Whether to interpolate style layers or mix.
        :param conditional: Whether to use conditional generation (only available for models trained with conditions).
        :param batch_size: The maximum batch size used to generate images.
        :raises ValueError: If `manipulation_mode` or `noise_mode` is not supported.
        """
        """Set constants."""
        self._interpolate = interpolate
        self._device = device
        # Dimensions of w -> we only want to mix style layers.
        self._mix_dims = torch.arange(*mix_dims, device=self._device)

        if noise_mode in ["random", "const"]:
            self._noise_mode = noise_mode
        else:
            raise ValueError(f"Unknown noise mode: {noise_mode}")

        """Load or set the generator network."""
        self._generator = (
            generator if isinstance(generator, torch.nn.Module) else load_stylegan(generator)
        )
        self._generator.to(self._device)
        self._has_input_transform = hasattr(self._generator.synthesis, "input") and hasattr(
            self._generator.synthesis.input, "transform"
        )
        self._conditional = conditional
        self._batch_size = batch_size

    def manipulate(
        self,
        candidates: MixCandidateList,
        cond: NDArray,
        weights: NDArray,
    ) -> Tensor:
        """
        Generate data using style mixing or interpolation.

        This function is heavily inspired by the Renderer class of the original StyleGANv3 codebase.

        :param candidates: The candidates used for style-mixing.
        :param cond: The manipulation conditions (layer combinations) (batch x genome_size).
        :param weights: The weights for manipulating layers (batch x genome_size).
        :returns: The generated image (C x H x W) as a float with range [0, 255].
        """
        assert (
            cond.shape == weights.shape
        ), "Error: The manipulation condition and weights have to be of same shape."
        assert len(self._mix_dims) == len(
            cond[1]
        ), f"Error SMX condition array is not the same size as the mix dimensions ({len(self._mix_dims)} vs {len(cond[1])}). This might be due to a mismatch in genome size."

        if not self._interpolate:
            weights = weights.astype(np.int_)
        weights = torch.as_tensor(weights, device=self._device, dtype=torch.float32)[..., None]

        if self._has_input_transform:
            m = np.eye(3)
            self._generator.synthesis.input.transform.copy_(torch.from_numpy(m))

        w_avg = self._generator.mapping.w_avg
        w_avg = w_avg if w_avg.ndim == 1 else w_avg.mean(dim=tuple(range(w_avg.ndim - 1)))
        wn_tensors = torch.vstack(candidates.wn_candidates.w_tensors) - w_avg
        wn_tensors = wn_tensors.repeat(cond.shape[0], *([1] * (wn_tensors.ndim - 1)))

        """Get w0 vector."""
        w0_weights = (candidates.w0_candidates.weights,)
        assert all(
            (sum(w) == 1 for w in w0_weights)
        ), f"Error: w0 weight do not sum up to 1: {w0_weights}."
        w0_weight_tensor = torch.as_tensor(w0_weights, device=self._device)[:, None, None]
        w0_tensors = torch.stack(candidates.w0_candidates.w_tensors) - w_avg
        w0 = (w0_tensors * w0_weight_tensor).sum(dim=0)  # Initialize base using w0 seeds.
        w0 = w0.repeat(cond.shape[0], *([1] * (w0.ndim - 1)))

        """Here we do style mixing on the final latent-vector w."""
        w = torch.zeros_like(w0, device=self._device, dtype=torch.float32)

        comp1 = wn_tensors[cond, self._mix_dims, :] * weights
        comp2 = w0[:, self._mix_dims] * (1 - weights)

        w[:, self._mix_dims] += comp1 + comp2
        w += w_avg

        imgs = self.synthesize(w)
        return imgs

    def get_w(self, seed: int, class_idx: int, batch_size: int = 1) -> Tensor:
        """
        Generate w vector.

        :param seed: The seed to generate.
        :param class_idx: The label.
        :param batch_size: How many examples should be generated.
        :returns: The w vector.
        """
        torch.manual_seed(seed)
        # Generate latent vectors
        z = torch.randn(size=[batch_size, self._generator.z_dim], device=self._device)
        # Set class conditional vector, if conditional sampling is used and allowed.
        if self._generator.c_dim != 0:
            c = torch.zeros(size=[batch_size, self._generator.c_dim], device=self._device)
            c[:, class_idx] = 1 if self._conditional else 0
        else:
            c = None
        w = self._generator.mapping(
            z=z, c=c, truncation_psi=self.trunc_psi, truncation_cutoff=self.trunc_cutoff
        )
        return w

    def synthesize(self, w: Tensor) -> Tensor:
        """
        Get a generated image from the w Vector.

        :param w: The w vector for generation (batch x w_dim).
        :returns: The generated image (batch x img_dim).
        """
        all_imgs = []
        batch_size = self._batch_size or w.shape[0]
        for i in range(0, w.shape[0], batch_size):
            w_batch = w[i : i + batch_size]
            out, _ = self._run_synthesis_net(
                self._generator.synthesis, w_batch, noise_mode=self._noise_mode, force_fp32=False
            )
            imgs = self._transform_images_output(out)
            all_imgs.append(imgs)

        return torch.cat(all_imgs, dim=0)

    @staticmethod
    def _transform_images_output(images: Tensor, full_range: bool = False) -> Tensor:
        """
        Transform generated image output.

        :param images: The images to be transformed.
        :param full_range: If the image should be casted to range [0,255] or [0,1].
        :returns: The transformed image.
        """
        images = images.to(torch.float32)  # B x C x W x H
        # Normalize color range.
        images /= images.norm(float("inf"), dim=[2, 3], keepdim=True).clip(1e-8, 1e8)
        images = (
            (images * 127.5 + 128).clamp(0, 255) if full_range else ((images + 1) / 2).clamp(0, 1)
        )
        return images

    # ------------------ Copied from https://github.com/NVlabs/stylegan3/blob/main/viz/renderer.py -----------------------
    @staticmethod
    def _run_synthesis_net(net, *args, capture_layer=None, **kwargs):  # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [
                out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]
            ]
            for idx, out in enumerate(outputs):
                if out.ndim == 5:  # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == "":
                    name = "output"
                if len(outputs) > 1:
                    name += f":{idx}"
                if name in unique_names:
                    suffix = 2
                    while f"{name}_{suffix}" in unique_names:
                        suffix += 1
                    name += f"_{suffix}"
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split(".")[-1]
                layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
        try:
            out = net(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for hook in hooks:
            hook.remove()
        return out, layers

    @staticmethod
    def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
        _N, _C, H, W = x.shape
        mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

        # Construct filter.
        f = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
        assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
        p = f.shape[0] // 2

        # Construct sampling grid.
        theta = mat.inverse()
        theta[:2, 2] *= 2
        theta[0, 2] += 1 / up / W
        theta[1, 2] += 1 / up / H
        theta[0, :] *= W / (W + p / up * 2)
        theta[1, :] *= H / (H + p / up * 2)
        theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
        g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

        # Resample image.
        y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
        z = torch.nn.functional.grid_sample(
            y, g, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Form mask.
        m = torch.zeros_like(y)
        c = p * 2 + 1
        m[:, :, c:-c, c:-c] = 1
        m = torch.nn.functional.grid_sample(
            m, g, mode="nearest", padding_mode="zeros", align_corners=False
        )
        return z, m


def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(
        1 - aflt * up
    )
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0, 1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0, 2], keepdim=True) / (up**2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f


def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float("inf"))
    return torch.where(y < 1e-30, torch.ones_like(x), z)


def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))


class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out
