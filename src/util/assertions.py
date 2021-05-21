from typing import Union, NoReturn, Optional

import numpy as np
import torch


def assert_dim(x: Union[np.ndarray, torch.Tensor], expected_dim: int, name: str) -> NoReturn:
    assert len(x.shape) == expected_dim, f"{name} has to be {expected_dim}-dimensional"


def assert_axis_length(x: Union[np.ndarray, torch.Tensor], axis: int, expected_length: int, name: str) -> NoReturn:
    assert x.shape[axis] == expected_length, f"axis {axis} of {name} must have length {expected_length}"


def assert_same_axis_length(x1: Union[np.ndarray, torch.Tensor], x2: Union[np.ndarray, torch.Tensor], axis1: int, name1: str, name2: str, axis2: Optional[int] = None) -> NoReturn:
    if axis2 is None:
        axis2 = axis1
    assert x1.shape[axis1] == x2.shape[axis2], f"{name1} and {name2} must have same axis lengths for axis {axis1} and {axis2}, respectively"


def assert_positive(n: Union[int, float], name: str) -> NoReturn:
    assert n > 0, f"{name} has to be positive"
