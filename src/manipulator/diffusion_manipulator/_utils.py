import os
from typing import Optional

import torch


def prepare_cuda(device: Optional[torch.device], require_grad: bool) -> torch.device:
    """
    Prepare optimized CUDA environment.

    :param device: The torch device to use if applicable.
    :param require_grad: If true, require grad.
    :returns: The torch device to use.
    """
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available(), "No GPU available please check your setup."

    torch.set_grad_enabled(require_grad)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional CUDA optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return device
