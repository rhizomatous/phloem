"""device selection utilities."""

import torch


def resolve_device(device: str) -> str:
    """resolve 'auto' to the best available device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
