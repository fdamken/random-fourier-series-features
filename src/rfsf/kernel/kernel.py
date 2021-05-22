from abc import ABC, abstractmethod

import torch

from rfsf.util.assertions import assert_dim, assert_same_axis_length


class Kernel(ABC):
    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        assert_dim(p, 2, "p")
        assert_dim(q, 2, "q")
        assert_same_axis_length(p, q, 1, "p", "q")

        result = self.forward(p, q)

        assert_dim(result, 2, "result")
        assert_same_axis_length(result, p, axis1=0, axis2=0, name1="result", name2="p")
        assert_same_axis_length(result, q, axis1=1, axis2=0, name1="result", name2="q")
        return result

    @abstractmethod
    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()  # pragma: no cover
