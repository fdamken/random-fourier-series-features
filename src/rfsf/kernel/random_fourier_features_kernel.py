import numpy as np
import torch

from rfsf.kernel.degenerate_kernel import DegenerateKernel
from rfsf.util.assertions import assert_axis_length, assert_positive


class RandomFourierFeaturesKernel(DegenerateKernel):
    """
    Implementation of Random Fourier Features (RFFs) which can be used to approximate the Squared Exponential (SE)
    kernel.

    .. seealso::
        A. Rahimi and B. Recht. "Random Features for Large-Scale Kernel Machines.", NIPS, 2007
    """

    def __init__(self, input_dim: int, num_features: int, length_scale: torch.Tensor = None, device: torch.device = None):
        """
        Constructor.

        :param input_dim: dimensionality of the input space
        :param num_features: number of features to use; the higher the number of features, the better the approximation
                             of the SE kernel
        :param length_scale: length scale to resemble; approximately equivalent to the length scale of the SE kernel
        """

        super().__init__(device=device)

        assert_positive(input_dim, "input_dim")
        assert_positive(num_features, "num_features")
        if length_scale is None:
            length_scale = torch.tensor(1.0, requires_grad=True)
        assert_positive(length_scale.item(), "length_scale")  # Implies checking that n is a scalar.

        self._input_dim = input_dim
        self._num_features = num_features
        self._length_scale = torch.nn.Parameter(length_scale, requires_grad=length_scale.requires_grad).to(device=self.device)

        weight_distribution = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        bias_distribution = torch.distributions.Uniform(0.0, 2 * np.pi)
        self._weights = weight_distribution.sample((self._num_features,)).to(device=self.device)
        self._biases = bias_distribution.sample((self._num_features,)).to(device=self.device)

        self.register_parameters(self._length_scale)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the random fourier features."""
        assert_axis_length(x, 1, self._input_dim, "x")
        return np.sqrt(2 / self._num_features) * torch.cos(x @ self._weights.T / self._length_scale + self._biases)
