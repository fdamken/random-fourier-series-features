from typing import NoReturn, Optional, Union

import numpy as np
import torch


def assert_dim(x: Union[np.ndarray, torch.Tensor], expected_dim: int, name: str) -> NoReturn:
    """Asserts that the given array/tensor `x` has dimensionality `expected_dim`, i.e., the length of the shape is equal
    to `expected_dim`. `name` is the name of the variable that is checked and is used for the error message if the
    assertion fails."""
    assert len(x.shape) == expected_dim, f"{name} has to be {expected_dim}-dimensional"


def assert_scalar(x: Union[np.ndarray, torch.Tensor], name: str) -> NoReturn:
    """Asserts that the given array/tensor `x` is a scalar, i.e., it has a zero-length shape. `name` is the name of the
       variable that is checked and is used for the error message if the assertion fails."""
    assert_dim(x, 0, name)


def assert_axis_length(x: Union[np.ndarray, torch.Tensor], axis: int, expected_length: int, name: str) -> NoReturn:
    """Asserts that axis `axis` of the array/tensor `x` has length `expected_length`. `name` is the name of the variable
    that is checked and is used for the error message if the assertion fails."""
    assert x.shape[axis] == expected_length, f"axis {axis} of {name} must have length {expected_length}"


def assert_same_axis_length(x1: Union[np.ndarray, torch.Tensor], x2: Union[np.ndarray, torch.Tensor], axis1: int, name1: str, name2: str, axis2: Optional[int] = None) -> NoReturn:
    """Asserts that `axis1` of `x1` and `axis2` of `x2` have the same length. If `axis2` is not given, `axis1` is used
    for both arrays/tensors. `name1` and `name2` are the names of the variables that are checked and are used for the
    error message if the assertion fails."""
    if axis2 is None:
        axis2 = axis1
    assert x1.shape[axis1] == x2.shape[axis2], f"{name1} and {name2} must have same axis lengths for axis {axis1} and {axis2}, respectively"


def assert_positive(n: Union[int, float, np.ndarray, torch.Tensor], name: str) -> NoReturn:
    """Asserts that `n` is positive. `name` is the name of the variable that is checked and is used for the error
    message if the assertion fails. If `n` is an array/tensor, it has to be a scalar according to
    :py:meth:`.is_scalar`."""
    if isinstance(n, (np.ndarray, torch.Tensor)):
        assert_scalar(n, name)
        n = n.item()
    assert n > 0, f"{name} has to be positive"
