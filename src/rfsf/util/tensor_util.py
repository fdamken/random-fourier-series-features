from typing import Callable, NoReturn, Union

import numpy as np
import torch


# Type alias to accept both NumPy arrays and Torch tensors.
NdTensor = Union[np.ndarray, torch.Tensor]


def is_numpy(val: NdTensor) -> bool:
    """Checks if the given :py:class:`NdTensor` is a NumPy array."""
    return isinstance(val, np.ndarray)


def is_torch(val: NdTensor) -> bool:
    """Checks if the given :py:class:`NdTensor` is a Torch tensor."""
    return isinstance(val, torch.Tensor)


def periodic(half_period: float) -> Callable[[Callable[[NdTensor], NdTensor]], NoReturn]:
    """
    Generates a decorator that can be used to make any function periodic by applying the modulus operator as follows:

    .. math::
        y = f([x + T]_{2T} - T)

    where :math:`T` is the `half_period`, :math:`f` is the wrapped function and :math:`x`, :math:`y` are the input and
    output values, respectively.

    :param half_period: half value of the period of the result for function; for sine this would be :math:`\\pi`
    :return: decorator to be applied on the function
    """

    def decorator(func: Callable[[NdTensor], NdTensor]) -> Callable[[NdTensor], NdTensor]:
        def wrapped(x: NdTensor):
            return func((x + half_period) % (2 * half_period) - half_period)

        return wrapped

    return decorator
