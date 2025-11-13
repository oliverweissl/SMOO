import os
from typing import Optional

import torch


def prepare_cuda(
    device: Optional[torch.device], require_grad: bool, precision: str = "medium"
) -> torch.device:
    """
    Prepare optimized CUDA environment.

    :param device: The torch device to use if applicable.
    :param require_grad: If true, require grad.
    :param precision: The precision to use in matmul [high, medium, low].
    :returns: The torch device to use.
    """
    assert torch.cuda.is_available(), "No GPU available please check your setup."
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.benchmark = True

    device = device or torch.device("cuda")

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)

    torch.set_grad_enabled(require_grad)
    return device
