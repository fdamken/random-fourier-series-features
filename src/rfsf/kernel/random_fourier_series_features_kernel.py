import numpy as np
import torch

from rfsf.kernel.degenerate_kernel import DegenerateKernel
from rfsf.util.assertions import assert_axis_length, assert_positive, assert_shape, assert_non_negative


class RandomFourierSeriesFeaturesKernel(DegenerateKernel):
    def __init__(self, input_dim: int, num_features: int, num_harmonics: int, length_scale: torch.Tensor = None, amplitudes_sqrt: torch.Tensor = None, phases: torch.Tensor = None,
                 device: torch.device = None):
        super().__init__(device=device)

        assert_positive(input_dim, "input_dim")
        assert_positive(num_features, "num_features")
        assert_non_negative(num_harmonics, "num_harmonics")
        if length_scale is None:
            length_scale = torch.tensor(1.0, requires_grad=True)
        assert_positive(length_scale.item(), "length_scale")  # Implies checking that n is a scalar.
        if amplitudes_sqrt is None:
            amplitudes_sqrt = torch.randn((num_harmonics + 1), requires_grad=True)
        if phases is None:
            phases = torch.randn((num_harmonics + 1,), requires_grad=True)
        assert_shape(amplitudes_sqrt, (num_harmonics + 1,), "amplitudes")
        assert_shape(phases, (num_harmonics + 1,), "phases")

        self._input_dim = input_dim
        self._num_features = num_features
        self._num_harmonics = num_harmonics
        self._length_scale = torch.nn.Parameter(length_scale, requires_grad=length_scale.requires_grad).to(device=self.device)
        self._amplitudes_sqrt = torch.nn.Parameter(amplitudes_sqrt, requires_grad=amplitudes_sqrt.requires_grad).to(device=self.device)
        self._phases = torch.nn.Parameter(phases, requires_grad=phases.requires_grad).to(device=self.device)

        weight_distribution = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        bias_distribution = torch.distributions.Uniform(0.0, 2 * np.pi)
        self._weights = weight_distribution.sample((self._num_features,)).to(device=self.device)
        self._biases = bias_distribution.sample((self._num_features,)).to(device=self.device)

        self.register_parameters(self._length_scale, self._amplitudes_sqrt, self._phases)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        assert_axis_length(x, 1, self._input_dim, "x")

        biases = self._biases.unsqueeze(dim=0).unsqueeze(dim=2)
        phases = self._phases.unsqueeze(dim=0).unsqueeze(dim=1)

        normalization = np.sqrt(2 / self._num_features)
        weighted_inputs = (x @ self._weights.T).unsqueeze(dim=2)
        harmonic_multipliers = (torch.arange(self._num_harmonics + 1, device=self.device)).unsqueeze(dim=0).unsqueeze(dim=1)
        harmonics_activations = harmonic_multipliers * weighted_inputs + biases + phases
        harmonics = self._amplitudes * torch.cos(harmonics_activations)
        return normalization * harmonics.sum(dim=2) / self._amplitudes.sum()

    @property
    def _amplitudes(self) -> torch.Tensor:
        return self._amplitudes_sqrt ** 2
