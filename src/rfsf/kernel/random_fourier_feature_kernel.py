from typing import Optional, Tuple

import numpy as np
import torch

from rfsf.kernel.feature_based_kernel import FeatureBasedKernel
from rfsf.util.assertions import assert_axis_length, assert_positive


class RandomFourierFeatureKernel(FeatureBasedKernel[Tuple[torch.Tensor, torch.Tensor]]):  # pylint: disable=unsubscriptable-object
    def __init__(self, input_dim: int, num_features: int, length_scale: float = 1.0, weight_distribution: torch.distributions.Distribution = None):
        assert_positive(input_dim, "input_dim")
        assert_positive(num_features, "num_features")
        assert_positive(length_scale, "length_scale")
        if weight_distribution is None:
            weight_distribution = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        assert weight_distribution.sample().shape == (input_dim,), "axis length of samples from the weight_distribution must match the input dimensionality"

        self._input_dim = input_dim
        self._num_features = num_features
        self._length_scale = length_scale
        self._weight_distribution = weight_distribution
        self._bias_distribution = torch.distributions.Uniform(0.0, 2 * np.pi)

    def forward_features(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert_axis_length(x, 1, self._input_dim, "x")

        if state is None:
            weights = self._weight_distribution.sample((self._num_features,))
            bias = self._bias_distribution.sample((self._num_features,))
        else:
            weights, bias = state
        return np.sqrt(2 / self._num_features) * np.cos(self._length_scale * x @ weights.T + bias), (weights, bias)
