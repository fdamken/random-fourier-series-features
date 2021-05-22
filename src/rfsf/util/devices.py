import torch


def cuda() -> torch.device:
    """Gets the torch CUDA device that is currently active."""
    assert torch.cuda.is_available(), "CUDA is not available"
    return torch.device("cuda", index=torch.cuda.current_device())


def cpu() -> torch.device:
    """Gets the torch CPU device-"""
    return torch.device("cpu")
