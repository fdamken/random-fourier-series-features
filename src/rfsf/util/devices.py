import torch


def cuda():
    return torch.device("cuda", index=torch.cuda.current_device())


def cpu():
    return torch.device("cpu")
