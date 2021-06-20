import codecs
import pickle
from typing import Any, Callable, NoReturn, TypeVar, Union

import numpy as np
import torch


T = TypeVar("T", dict, list, np.ndarray, torch.Tensor)  # pylint: disable=invalid-name

# Type alias to accept both NumPy arrays and Torch tensors.
NdTensor = Union[np.ndarray, torch.Tensor]


def is_numpy(val: NdTensor) -> bool:
    """Checks if the given :py:class:`NdTensor` is a NumPy array."""
    return isinstance(val, np.ndarray)


def is_torch(val: NdTensor) -> bool:
    """Checks if the given :py:class:`NdTensor` is a Torch tensor."""
    return isinstance(val, torch.Tensor)


def to_numpy(val: NdTensor, *, force_detach: bool = False, copy: bool = False) -> np.ndarray:
    """
    Converts the given tensor (which can also be a NumPy array) into a NumPy array and copies it to the CPU. If it
    already is a NumPy array, it is not converted (this method is idempotent). If the Torch tensor `requires_grad`, it
    is detached first. Additionally, if `force_detach` or `copy` is `True`, it is also detached. If `copy` is `True`,
    the converted NumPy array is copied using the :py:meth:`numpy.ndarray.copy` method if the
    :py:attr:`numpy.ndarray.base` is not `None`.

    :param val: tensor to convert
    :param force_detach: whether to always detach the tensor before copying it to the CPU, ignoring whether it
                         `requires_grad`
    :param copy: whether to copy the NumPy array after conversion (implies `force_detach`); only applied if the `base`
                 of the array is not `None`
    :return: the numpy array with same shape as `val`; the underlying data type is handled automatically by PyTorch's
             :py:meth:`torch.Tensor.numpy` method
    """
    force_detach = force_detach or copy
    if not is_numpy(val):
        if val.requires_grad or force_detach or copy:
            val = val.detach()
        val = val.cpu().numpy()
    if copy and val.base is not None:
        val = val.copy()
    return val


def pickle_str(obj: Any) -> str:
    """
    Pickles the given object using :py:meth:`pickle.dumps` and base64-encodes the result. Can be unpickled using
    :py:meth:`.unpickle_str`.

    :param obj: object to pickle
    :return: base64-representation of the pickled object
    """
    return codecs.encode(pickle.dumps(obj), "base64").decode()


def unpickle_str(obj: str) -> Any:
    """
    Unpickles the given base64-representation of a pickle object as produced by :py:meth:`.pickle_str`.

    :param obj: base64-representation of the pickled object
    :return: unpickled object
    """
    return pickle.loads(codecs.decode(obj.encode(), "base64"))


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
