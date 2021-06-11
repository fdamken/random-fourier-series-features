import warnings

import torch


def cuda() -> torch.device:
    """Gets the torch CUDA device that is currently active."""
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, using CPU!")
        return torch.device("cpu")
    return torch.device("cuda", index=torch.cuda.current_device())


def cpu() -> torch.device:
    """Gets the torch CPU device-"""
    return torch.device("cpu")
