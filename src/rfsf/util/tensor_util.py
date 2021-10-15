import codecs
import pickle
import warnings
from itertools import product
from typing import Any, Callable, Iterator, List, NoReturn, Tuple, TypeVar, Union

import numpy as np
import torch
from numpy.linalg import LinAlgError
from torch import nn


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


def process_as_numpy_array(tensor: torch.Tensor, func: Callable[[np.ndarray], np.ndarray]) -> torch.Tensor:
    """
    Invokes the given callable `func` with a detached version of `tensor` (using :py:meth:`to_numpy`) converted to a
    NumPy array. The resulting NumPy array is than converted back to a Torch tensor using :py:meth:`torch.from_numpy`.
    The tensor is copied back to the original device `tensor` is on.

    :param tensor: tensor to process
    :param func: callable to invoke with the given tensor converted to a NumPy array
    :return: resulting NumPy array converted to a Torch tensor
    """
    return torch.from_numpy(func(to_numpy(tensor))).to(tensor.device)


def split_parameter_groups(model: nn.Module) -> Tuple[List[str], List[torch.nn.Parameter]]:
    """
    Gets the parameters of the given `model` and, if they require taking a gradient, adds them to the list of learnable
    parameters. The names of the parameters groups are also extracted.

    :param model: model to extract the parameters from
    :return: tuple `(parameter_group_names, opt_parameters)`; the names of the parameter groups and the learnable
             parameters such that the `i`-th element of the parameter group names corresponds to the `i`-th parameter
    """
    parameter_group_names, opt_parameters = [], []
    for name, params in model.named_parameters():
        if params.requires_grad:
            parameter_group_names.append(name)
            opt_parameters.append(params)
    return parameter_group_names, opt_parameters


def apply_parameter_name_selector(names: List[str], selector: List[str]) -> List[str]:
    """
    Applies the given `selector` to the given list of `names`. A name is included in the result if it is contained in
    the given `selector` and a name is removed from the result if it is contained in the given `selector` and is
    preceded by an exclamation mark (`!`). The special name `all` includes all `names` in the result. Excludes take
    precedence over includes.

    :param names: all available names
    :param selector: selector to apply
    :return: selected names according to the above rules; contains no duplicated
    """
    result = []
    negations = []
    for filt in selector:
        if filt == "all":
            result += names
        elif filt.startswith("!"):
            negations.append(filt[1:])
        elif filt in names:
            result.append(filt)
        else:
            assert False, f"unknown name {filt!r}"
    for filt in negations:
        if filt in names:
            result.remove(filt)
        else:
            assert False, f"unknown name {filt!r}"
    return list(set(result))


def gen_index_iterator(val: NdTensor) -> Iterator[tuple]:
    """
    Creates an iterator over the indices of the given tensor (which can also be a NumPy array), over all dimensions.
    The resulting iterator is a list of tuples each corresponding to a single element. For example, the elements of the
    iterator for an array with shape `(2, 3)` would be `(0, 0)`, `(0, 1)`, `(0, 2)`, `(1, 0)`, `(1, 1)`, and `(1, 2)`.

    :param val: tensor to iterate over
    :return: iterator with the index tuples as described before
    """
    assert len(val.shape) > 0, "val has to have at least one dimension"
    indices = [range(dim_length) for dim_length in val.shape]
    return product(*indices)


def make_positive_definite(val: NdTensor, *, warn_on_jitter: bool = True, jitter_exponent_lo: int = -10, jitter_exponent_up: int = 0) -> NdTensor:
    """
    Makes the given two-dimensional quadratic tensor `val` positive definite by adding jittering on the diagonal. Tries
    to start with no jittering and subsequently adds jittering starting from `10 ** jitter_exponent_lo` up to
    `10 ** jitter_exponent_up` (in equidistant steps of `1` in the exponent).

    :param val: matrix to make positive definite
    :param warn_on_jitter: whether a `UserWarning` should be printed when jittering is applied
    :param jitter_exponent_lo: first jittering exponent value that is applied
    :param jitter_exponent_up: last jittering exponent to try
    :return: the jittered matrix
    :raises RuntimeError: if the matrix was not positive definite after adding jittering of `10 ** jitter_exponent_up`
    """

    assert len(val.shape) == 2, "val has to be two-dimensional"

    if is_numpy(val):
        eye = np.eye(val.shape[-1])
    elif is_torch(val):
        eye = torch.eye(val.shape[-1])
    else:
        assert False, "A is neither NumPy nor Torch tensor"

    val_jittered = val
    jitter_exponent = jitter_exponent_lo
    while jitter_exponent <= jitter_exponent_up:
        try:
            if is_numpy(val_jittered):
                np.linalg.cholesky(val_jittered)
            elif is_torch(val_jittered):
                torch.linalg.cholesky(val_jittered)
            else:
                assert False, "val_jittered is neither NumPy nor Torch tensor"
            break
        except (LinAlgError, RuntimeError) as ex:
            if isinstance(ex, LinAlgError) or "singular U." in getattr(ex, "args", [""])[0]:
                if warn_on_jitter:
                    warnings.warn(f"Inverse transformed covariance is singular, adding jitter of 10e{jitter_exponent}.", UserWarning)
                val_jittered = val + 10 ** jitter_exponent * eye
                jitter_exponent += 1
            else:
                raise ex
    else:  # Invoke iff the above loop does not exit with the break but ends regularly.
        raise RuntimeError(f"A is not positive definite after adding jittering up to 10e{jitter_exponent_up}.")
    return val_jittered
