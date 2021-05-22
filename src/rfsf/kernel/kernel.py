from abc import ABC, abstractmethod
from typing import List, NoReturn

import torch

from rfsf.util.assertions import assert_dim, assert_same_axis_length


class Kernel(ABC):
    """
    Abstract base class for a kernel, also known as a *covariance function*. Every kernel implements the `__call__`
    method, i.e., to get the value of a kernel, it has to be invoked like a function.
    """

    def __init__(self):
        self._parameters: List[torch.Tensor] = []

    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Computes the kernel between the given two tensors. The tensors are required to have a batch-dimension as the
        axis that can be used to compute the Gram matrix between different tensors.

        .. note::
            This method forwards the computation to :py:meth:`.forward`, which has to be overwritten by subclasses. Do
            not overwrite the :py:meth:`.__call__` method directly as it ensures that the in-/output tensors have the
            correct shapes!

        :param p: first argument to the tensor; shape `(N, k)`
        :param q: second argument to the kernel; shape `(M, k)`
        :return: the value of the kernel evaluated for all combinations of the different tensor entries, i.e., the Gram
                 matrix of the two arguments; shape `(N, M)`
        """
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
        """
        Carries out the computation of the kernel.

        .. note::
            Do not invoke this method directly, but rather invoke :py:meth:`.__call__`.

        .. seealso:: methods :py:meth:`.__call__`

        :param p: shape `(N, k)`; first argument to the tensor
        :param q: shape `(M, k)`; second argument to the kernel
        :return: shape `(N, M)`; the value of the kernel evaluated for all combinations of the different tensor entries,
                 i.e., the Gram matrix of the two arguments
        """
        raise NotImplementedError()  # pragma: no cover

    def register_parameters(self, *params: torch.Tensor) -> NoReturn:
        """
        Tests for all of the given tensors whether they have `requires_grad` set to `True` and if so, treats them as a
        hyperparameter such that they will be returned by :py:meth:`.parameters`.

        :param params: parameters to register
        """
        for param in params:
            if param.requires_grad:
                self._parameters.append(param)

    @property
    def parameters(self) -> List[torch.Tensor]:
        """Gets the learnable parameters, i.e., hyperparameters, of this kernel. All of the parameters have
           `requires_grad` set to `True`."""
        return self._parameters
