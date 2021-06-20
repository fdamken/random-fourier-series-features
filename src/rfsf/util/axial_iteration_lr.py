from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class AxialIterationLR(LambdaLR):
    """
    This is a learning rate scheduler that allows switching between the parameters to optimize. For example: for three
    parameters `a`, `b`, and `c`, where the first two are iterated, this learning rate scheduler would zero out the
    learning rate of `a`, then of `b`, then `a` again. Hence, in the first iteration the parameters `a` and `c` are
    optimized, then `a` and `b`, then `a` and `c`, and so on.

    This can be used to smooth out the learning process, e.g., if the parameters have a multiplicative dependency.
    """

    def __init__(self, optimizer: Optimizer, parameter_group_names: List[str], axial_iteration_over: List[str]):
        """
        Constructor.

        :param optimizer: optimizer that is used; the order of the parameter groups have to match the parameters names
                          in `parameter_group_names`
        :param parameter_group_names: names of the parameter groups as referred to by `axial_iteration_over`; must match
                                      in length the number of parameter groups of the optimizer
        :param axial_iteration_over: names of the parameters to axial-iterate over; must be a sub-list of
                                     `parameter_group_names`
        """
        assert len(optimizer.param_groups) == len(parameter_group_names), "number of the optimizer's parameter groups has to match the length of the given parameter group names"
        assert all(name in parameter_group_names for name in axial_iteration_over), f"all items of {axial_iteration_over=} must be in {parameter_group_names=}"
        total_axial_dim = len(axial_iteration_over)
        lambdas = []
        for parameter_group_name in parameter_group_names:
            if parameter_group_name in axial_iteration_over:
                lambdas.append(lambda epoch: int(epoch % total_axial_dim == 0))
            else:
                lambdas.append(lambda _: 1)
        super().__init__(optimizer, lambdas)
