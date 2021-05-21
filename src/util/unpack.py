from typing import TypeVar, Union, Tuple

T = TypeVar("T")  # pylint: disable=invalid-name
V = TypeVar("V")  # pylint: disable=invalid-name


def unpack(val: Union[T, Tuple[T, V]], default: V) -> Tuple[T, V]:
    if isinstance(val, tuple):
        return val
    return val, default
