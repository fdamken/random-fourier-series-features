from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ConstantLR(LambdaLR):
    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer, lambda _: 1)
