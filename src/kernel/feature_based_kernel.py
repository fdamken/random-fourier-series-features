from abc import abstractmethod
from typing import Optional, Tuple, TypeVar, Generic, Union

import torch

from kernel.kernel import Kernel
from util.assertions import assert_dim
from util.unpack import unpack

T = TypeVar("T")  # pylint: disable=invalid-name


class FeatureBasedKernel(Kernel, Generic[T]):
    def features(self, x: torch.Tensor, state: Optional[T]) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[T]]]:
        assert_dim(x, 2, "x")
        return self.forward_features(x, state)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        phi_p, state = unpack(self.features(p, None), None)
        phi_q, _ = unpack(self.features(q, state), None)
        return phi_p @ phi_q.T

    @abstractmethod
    def forward_features(self, x: torch.Tensor, state: Optional[T]) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[T]]]:
        raise NotImplementedError()  # pragma: no cover
