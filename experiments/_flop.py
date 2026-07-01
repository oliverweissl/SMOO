from __future__ import annotations

import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from torch import Tensor


def count_flops(model: nn.Module, inputs, verbose: bool = False) -> int:
    """Count total FLOPs via fvcore. Custom CUDA ops count as 0.

    :param model: Module to profile.
    :param inputs: Inputs passed to fvcore JIT trace.
    :param verbose: Print warnings for unsupported/uncalled ops.
    :returns: Total FLOPs.
    :rtype: int
    """
    fa = FlopCountAnalysis(model, inputs)
    fa.unsupported_ops_warnings(verbose)
    fa.uncalled_modules_warnings(verbose)
    return fa.total()


def count_flops_and_trainable(
    model: nn.Module, inputs, verbose: bool = False
) -> tuple[int, int]:
    """Single fvcore trace returning total and trainable-only FLOPs.

    Avoids tracing the model twice — critical for large diffusion models.
    Only leaf modules are summed to avoid double-counting parent + child FLOPs.

    :param model: Module to profile.
    :param inputs: Inputs passed to fvcore JIT trace.
    :param verbose: Print warnings for unsupported/uncalled ops.
    :returns: ``(total_flops, trainable_only_flops)``
    :rtype: tuple[int, int]
    """
    fa = FlopCountAnalysis(model, inputs)
    fa.unsupported_ops_warnings(verbose)
    fa.uncalled_modules_warnings(verbose)
    total  = fa.total()
    by_mod = fa.by_module()
    trainable = sum(
        by_mod.get(name, 0)
        for name, module in model.named_modules()
        if not list(module.children())
        and any(p.requires_grad for p in module.parameters(recurse=False))
    )
    return total, trainable


# ---------------------------------------------------------------------------
# StyleGAN custom CUDA op handlers
#
# upfirdn2d  : 2D FIR upsample/downsample — 4-tap filter → 16 muls+adds ≈ 32 FLOPs/elem
# bias_act   : bias + leaky relu → 2 FLOPs/elem
# filtered_lrelu: FIR filter + activation → 32 + 2 = 34 FLOPs/elem
# ---------------------------------------------------------------------------

def _out_numel(outputs) -> int:
    out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    return int(out.numel()) if hasattr(out, "numel") else 0


_STYLEGAN_HANDLERS: dict = {
    "upfirdn2d":      lambda i, o: _out_numel(o) * 32,
    "bias_act":       lambda i, o: _out_numel(o) * 2,
    "filtered_lrelu": lambda i, o: _out_numel(o) * 34,
}


def count_flops_stylegan(model: nn.Module, inputs, verbose: bool = False) -> int:
    """Count FLOPs including estimated FLOPs for StyleGAN custom CUDA ops.

    Performs a dry run to discover op names, registers handlers, then counts.

    :param model: StyleGAN module to profile.
    :param inputs: Inputs passed to fvcore JIT trace.
    :param verbose: Print discovered and remaining uncounted ops.
    :returns: Total FLOPs including custom op estimates.
    :rtype: int
    """
    fa_discover = FlopCountAnalysis(model, inputs)
    fa_discover.unsupported_ops_warnings(verbose)
    fa_discover.uncalled_modules_warnings(verbose)
    fa_discover.total()
    found = fa_discover.unsupported_ops()

    if verbose and found:
        print("Custom ops found (pre-handler):")
        for op, cnt in found.items():
            print(f"  {op}: {cnt}×")

    fa = FlopCountAnalysis(model, inputs)
    fa.unsupported_ops_warnings(verbose)
    fa.uncalled_modules_warnings(verbose)
    for op_name in found:
        op_lower = op_name.lower()
        for keyword, handler in _STYLEGAN_HANDLERS.items():
            if keyword in op_lower:
                fa.set_op_handle(**{op_name: handler})
                if verbose:
                    print(f"  Registered handler for '{op_name}'")
                break

    total = fa.total()
    if verbose and (remaining := fa.unsupported_ops()):
        print("Still uncounted after handlers:", list(remaining.keys()))
    return total


# ---------------------------------------------------------------------------
# Module wrappers for FLOPs profiling
# ---------------------------------------------------------------------------

class SynthWrapper(nn.Module):
    """Wraps StyleGAN synthesis network for FLOPs profiling.

    :param synth: StyleGAN synthesis network.
    """

    def __init__(self, synth: nn.Module) -> None:
        super().__init__()
        self.s = synth

    def forward(self, w: Tensor) -> Tensor:
        return self.s(w, noise_mode="const", force_fp32=False)


class _FlatEmbedder(nn.Module):
    """Registered-module replacement for ControlProjector's 1-D partial embedder.

    :param linear: Linear layer extracted from the partial.
    :param input_shape: Output spatial shape after the linear projection.
    """

    def __init__(self, linear: nn.Module, input_shape: tuple) -> None:
        super().__init__()
        self.linear = linear
        self.input_shape = input_shape

    def forward(self, control: Tensor) -> Tensor:
        b = control.size(0)
        return self.linear(control.view(b, -1)).view(b, *self.input_shape)


class _1dEmbedder(nn.Module):
    """Registered-module replacement for ControlProjector's 2-D partial embedder.

    Mirrors ``ControlProjector._1d_embedder``: Conv1d over channels then Linear
    over sequence length, followed by reshape to spatial ``input_shape``.

    :param conv: Conv1d extracted from the partial's ``module`` kwarg.
    :param resizer: Linear extracted from the partial's ``resizer`` kwarg.
    :param input_shape: Target spatial shape ``(C, H, W)``.
    """

    def __init__(self, conv: nn.Module, resizer: nn.Module, input_shape: tuple) -> None:
        super().__init__()
        self.conv = conv
        self.resizer = resizer
        self.input_shape = input_shape

    def forward(self, control: Tensor) -> Tensor:
        b = control.size(0)
        x = self.conv(control)    # (b, C_in, L) → (b, C_out, L)
        x = self.resizer(x)       # (b, C_out, L) → (b, C_out, H*W)
        return x.view(b, *self.input_shape)


def fix_control_projector(model: nn.Module) -> None:
    """Replace functools.partial embedders in ControlProjector with nn.Modules.

    ControlProjector uses partials for 1-D and 2-D controls; fvcore JIT-traces
    captured nn.Module keyword args as tensor constants. This replaces them
    in-place with registered modules so weights are proper parameters.

    - 1-D control (``module`` only) → ``_FlatEmbedder``
    - 2-D control (``module`` + ``resizer``) → ``_1dEmbedder``

    :param model: Model tree to patch (searched recursively via ``model.modules()``).
    """
    from functools import partial as _partial

    for module in model.modules():
        if hasattr(module, "embedder") and isinstance(module.embedder, _partial):
            kw = module.embedder.keywords
            has_mod = "module" in kw and isinstance(kw["module"], nn.Module)
            has_rsz = "resizer" in kw and isinstance(kw["resizer"], nn.Module)
            if has_mod and has_rsz:
                module.embedder = _1dEmbedder(kw["module"], kw["resizer"], module.input_shape)
            elif has_mod:
                module.embedder = _FlatEmbedder(kw["module"], module.input_shape)


class SiTOneStepWrapper(nn.Module):
    """Wraps one ``_denoise_step`` of SiTHyperNet for per-step FLOPs profiling.

    Profile once and multiply by ``n_steps``. Pass ``batch=2`` tensors when
    ``cfg > 1.0`` — CFG doubles the batch inside the full forward.

    :param hyper_net: SiTHyperNet instance.
    """

    def __init__(self, hyper_net: nn.Module) -> None:
        super().__init__()
        self.hyper_net = hyper_net
        self.hyper_net.use_checkpoints = False

    def forward(self, x: Tensor, t: Tensor, y_embed: Tensor, control: Tensor) -> Tensor:
        return self.hyper_net._denoise_step(x, t, y_embed, control)


class UNet2DOneStepWrapper(nn.Module):
    """Wraps one ``_diffusion_step`` of UNet2DHyperNet for per-step FLOPs profiling.

    Profile once and multiply by ``n_steps``.

    :param hyper_net: UNet2DHyperNet instance.
    """

    def __init__(self, hyper_net: nn.Module) -> None:
        super().__init__()
        self.hyper_net = hyper_net
        self.hyper_net.use_checkpoints = False

    def forward(self, x: Tensor, control: Tensor, t: Tensor) -> Tensor:
        return self.hyper_net._diffusion_step(x, control, t)


class SDCNOneStepWrapper(nn.Module):
    """Wraps one denoising step of SDCNHyperNet for per-step FLOPs profiling.

    Profile once and multiply by ``n_steps``. Pass batch=1 tensors;
    CFG doubling is handled internally.

    :param hyper_net: SDCNHyperNet instance.
    """

    def __init__(self, hyper_net: nn.Module) -> None:
        super().__init__()
        self.hyper_net = hyper_net
        self.hyper_net.use_checkpoints = False
        # xformers XFormersAttnProcessor uses a custom CUDA op that emits c10::SymInt
        # arguments, which fvcore's JIT tracer cannot handle. Switch both the ControlNet
        # and UNet to standard (math) attention before tracing.
        hyper_net._controlnet.disable_xformers_memory_efficient_attention()
        hyper_net._model.disable_xformers_memory_efficient_attention()
        fix_control_projector(hyper_net)

    def forward(
        self,
        xt: Tensor,
        y_embed: Tensor,
        t: Tensor,
        control: Tensor,
        control_signal: Tensor,
    ) -> Tensor:
        hn = self.hyper_net
        latents = torch.cat([xt] * 2) if hn._do_cfg else xt
        cn_mid, cn_down = hn._controlnet_forward(latents, latents, y_embed, t, control_signal)
        hy_mid, hy_down = hn._hynea_forward(latents, y_embed, t, control)
        mid_res = cn_mid + hy_mid
        down_res = [a + b for a, b in zip(cn_down, hy_down)]
        return hn._unet_forward(latents, y_embed, t, None, down_res, mid_res)


class VaeDecodeWrapper(nn.Module):
    """Wraps a HuggingFace VAE's decode method as an nn.Module for FLOPs profiling.

    :param vae: HuggingFace VAE module with a ``.decode()`` method.
    """

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, z: Tensor) -> Tensor:
        return self.vae.decode(z).sample


# ---------------------------------------------------------------------------
# Stable Diffusion inference FLOPs
# ---------------------------------------------------------------------------

def sd_inference_flops(
    base_model: str,
    n_steps: int,
    device: torch.device,
    lora_path: str | None = None,
    latent_shape: tuple = (1, 4, 64, 64),
) -> tuple[int, int, int]:
    """Compute FLOPs for one complete SD image generation.

    Formula: ``UNet_step_flops × n_steps + VAE_decode_flops``.
    LoRA changes weights only — architecture (and FLOPs) are identical to base.

    :param base_model: HuggingFace model ID or local path.
    :param n_steps: Number of denoising steps.
    :param device: Target device.
    :param lora_path: Optional LoRA checkpoint path.
    :param latent_shape: Latent tensor shape ``(B, C, H, W)``.
    :returns: ``(total_flops, unet_step_flops, vae_decode_flops)``
    :rtype: tuple[int, int, int]
    """
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float32, safety_checker=None
    ).to(device)
    if lora_path:
        pipe.load_lora_weights(lora_path)

    unet = pipe.unet.eval()
    vae  = pipe.vae.eval()
    dummy_latent  = torch.randn(*latent_shape, device=device)
    dummy_t       = torch.tensor([0], device=device)
    dummy_enc_hid = torch.randn(1, 77, unet.config.cross_attention_dim, device=device)

    flops_unet_step = count_flops(unet, (dummy_latent, dummy_t, dummy_enc_hid))
    flops_vae_dec   = count_flops(VaeDecodeWrapper(vae), torch.randn(*latent_shape, device=device))
    del pipe, unet, vae
    torch.cuda.empty_cache()
    return flops_unet_step * n_steps + flops_vae_dec, flops_unet_step, flops_vae_dec


# ---------------------------------------------------------------------------
# Per-config FLOPs cache (enables cross-environment runs)
# ---------------------------------------------------------------------------

def save_flops(name: str, fwd: int, eff: int | None = None) -> None:
    """Save FLOPs to ``_{name}.json`` next to the notebook.

    :param name: Config key (e.g. ``"mimicry_imagenet"``).
    :param fwd: Forward-pass FLOPs.
    :param eff: Effective training FLOPs; defaults to ``fwd`` if omitted.
    """
    with open(f"_{name}.json", "w") as f:
        json.dump({"fwd": fwd, "eff": eff if eff is not None else fwd}, f)


def load_flops(name: str) -> tuple[int | None, int | None]:
    """Load FLOPs from ``_{name}.json``.

    :param name: Config key matching a prior :func:`save_flops` call.
    :returns: ``(fwd, eff)`` or ``(None, None)`` if the file does not exist.
    :rtype: tuple[int | None, int | None]
    """
    path = f"_{name}.json"
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    return d["fwd"], d["eff"]


# ---------------------------------------------------------------------------
# Budget loading
# ---------------------------------------------------------------------------

def load_budgets(pattern: str, field: str = "budget_used", extra_fields: tuple = ()) -> np.ndarray:
    """Load budget values from stats JSON files matching a glob pattern.

    SMOO runs use ``budget_used``; GIFTBench runs use ``budget``.
    Pass ``extra_fields=("w0_trials", "wn_trials")`` for Mimicry to get true total evaluations.

    :param pattern: Glob pattern for stats JSON files.
    :param field: Primary key to extract from each JSON object.
    :param extra_fields: Additional keys to sum into the budget value.
    :returns: Array of budget values found across all matched files.
    :rtype: numpy.ndarray
    """
    budgets = []
    for f in glob(pattern):
        with open(f) as fh:
            d = json.load(fh)
            if field in d:
                budgets.append(d[field] + sum(d.get(ef, 0) for ef in extra_fields))
    return np.array(budgets, dtype=float)