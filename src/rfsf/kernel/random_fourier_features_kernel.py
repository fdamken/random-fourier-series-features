import numpy as np
import torch

from rfsf.kernel.feature_based_kernel import DegenerateKernel
from rfsf.util.assertions import assert_axis_length, assert_positive


class RandomFourierFeaturesKernel(DegenerateKernel):
    """
    Implementation of Random Fourier Features (RFFs) which can be used to approximate the Squared Exponential (SE)
    kernel.

    .. seealso::
        A. Rahimi and B. Recht. "Random Features for Large-Scale Kernel Machines.", NIPS, 2007
    """

    def __init__(self, input_dim: int, num_features: int, length_scale: float = 1.0, weight_distribution: torch.distributions.Distribution = None):
        """
        Constructor.

        :param input_dim: dimensionality of the input space
        :param num_features: number of features to use; the higher the number of features, the better the approximation
                             of the SE kernel
        :param length_scale: length scale to resemble; approximately equivalent to the length scale of the SE kernel
        :param weight_distribution: distributions to use to sample the weights from; defaults to a Gaussian with zero
                                    mean and unit covariance; samples have to have shape `(input_dim,)`
        """
        assert_positive(input_dim, "input_dim")
        assert_positive(num_features, "num_features")
        assert_positive(length_scale, "length_scale")
        if weight_distribution is None:
            weight_distribution = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        assert weight_distribution.sample().shape == (input_dim,), "axis length of samples from the weight_distribution must match the input dimensionality"

        self._input_dim = input_dim
        self._num_features = num_features
        self._length_scale = length_scale
        self._weights = weight_distribution.sample((self._num_features,))
        self._biases = torch.distributions.Uniform(0.0, 2 * np.pi).sample((self._num_features,))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the random fourier features."""
        assert_axis_length(x, 1, self._input_dim, "x")
        return np.sqrt(2 / self._num_features) * np.cos(x @ self._weights.T / self._length_scale + self._biases)
