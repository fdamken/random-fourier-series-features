from abc import abstractmethod

import torch

from rfsf.kernel.kernel import Kernel
from rfsf.util.assertions import assert_dim, assert_device


class DegenerateKernel(Kernel):
    """
    Abstract subclass of :py:class:`rfsf.kernel.kernel.Kernel` that is feature-based, i.e., it resembles a degenerate
    kernel of which the value is computed by explicitly lifting the input values into a feature space and computing the
    dot product of both feature vectors.
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the features for the given input data `x`.

        .. note::
            This method forwards the computation to :py:meth:`.forward_features`, which has to be overwritten by
            subclasses. Do not overwrite the :py:meth:`.features` method directly as it ensures that the in-/output
            tensors have the correct shapes!

        :param x: input data; shape `(N, k)`
        :return: features for the given input data; shape `(N, d)`, where `d` is the dimensionality of the feature space
        """
        assert_device(x, self.device, "x")
        assert_dim(x, 2, "x")
        return self.forward_features(x)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Computes the kernel value by taking the dot product of the computed features."""
        return self.features(p) @ self.features(q).T

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the features for the given input data `x`.

        .. note::
            Do not invoke this method directly, but rather invoke :py:meth:`.features` for computing the features and
            :py:meth:`.__call__` for computing the kernel values.

        .. seealso:: methods :py:meth:`.features`

        :param x: input data; shape `(N, k)`
        :return: features for the given input data; shape `(N, d)`, where `d` is the dimensionality of the feature space
        """
        raise NotImplementedError()  # pragma: no cover
