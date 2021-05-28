from typing import Callable, NoReturn, Union

import numpy as np
import torch


NdTensor = Union[np.ndarray, torch.Tensor]


def is_numpy(val: NdTensor) -> bool:
    return isinstance(val, np.ndarray)


def is_torch(val: NdTensor) -> bool:
    return isinstance(val, torch.Tensor)


def periodic(half_period: float) -> Callable[[Callable[[NdTensor], NdTensor]], NoReturn]:
    def decorator(func: Callable[[NdTensor], NdTensor]) -> Callable[[NdTensor], NdTensor]:
        def wrapped(x: NdTensor):
            return func((x + half_period) % (2 * half_period) - half_period)

        return wrapped

    return decorator
