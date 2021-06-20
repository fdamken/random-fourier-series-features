from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class AxialIterationLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, axial_iteration_over: List[str], parameter_group_names: List[str]):
        assert all(name in parameter_group_names for name in axial_iteration_over), f"all items of {axial_iteration_over=} must be in {parameter_group_names=}"
        total_axial_dim = len(axial_iteration_over)
        lambdas = []
        for parameter_group_name in parameter_group_names:
            if parameter_group_name in axial_iteration_over:
                lambdas.append(lambda epoch: int(epoch % total_axial_dim == 0))
            else:
                lambdas.append(lambda _: 1)
        super().__init__(optimizer, lambdas)
