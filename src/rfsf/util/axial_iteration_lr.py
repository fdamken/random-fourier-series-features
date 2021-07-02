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

    def __init__(self, optimizer: Optimizer, parameter_group_names: List[str], axial_iteration_over: List[str], epoch_inverse_scale: int = 1):
        """
        Constructor.

        :param optimizer: optimizer that is used; the order of the parameter groups have to match the parameters names
                          in `parameter_group_names`
        :param parameter_group_names: names of the parameter groups as referred to by `axial_iteration_over`; must match
                                      in length the number of parameter groups of the optimizer
        :param axial_iteration_over: names of the parameters to axial-iterate over; must be a sub-list of
                                     `parameter_group_names`
        :param epoch_inverse_scale: scaling that is inversely applied to the epoch, i.e., the epoch is divided by
                                    `epoch_inverse_scale`; can be used to "stretch out" the axial switching, e.g., an
                                    inverse epoch scale of `2` would cause two steps with parameter `a`, then two with
                                    `b`, then two with `a`, and so on
        """
        assert len(optimizer.param_groups) == len(parameter_group_names), "number of the optimizer's parameter groups has to match the length of the given parameter group names"
        assert all(name in parameter_group_names for name in axial_iteration_over), f"all items of {axial_iteration_over=} must be in {parameter_group_names=}"
        total_axial_dim = len(axial_iteration_over)
        lambdas = []
        k = 0
        for parameter_group_name in parameter_group_names:
            if parameter_group_name in axial_iteration_over:
                # Use a higher-order function to prevent python from applying the last value of k to every lambda.
                lambdas.append((lambda _k: lambda epoch: int((epoch // epoch_inverse_scale) % total_axial_dim == _k))(k))
                k += 1
            else:
                lambdas.append(lambda _: 1)
        super().__init__(optimizer, lambdas)
