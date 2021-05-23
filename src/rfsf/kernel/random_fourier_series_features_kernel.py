import numpy as np
import torch

from rfsf.kernel.degenerate_kernel import DegenerateKernel
from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer
from rfsf.util.assertions import assert_axis_length, assert_positive


class RandomFourierSeriesFeaturesKernel(DegenerateKernel):
    """
    Implementation of Random Fourier Series Features (RFSFs).

    .. warning::
        This class is part of ongoing research!
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        fourier_series_init: FourierSeriesInitializer = None,
        length_scale: torch.Tensor = None,
        device: torch.device = None,
    ):
        """
        Constructor.

        .. note::
            This class works with the square-root of the amplitudes and squares them if needed to ensure that not non-
            positive amplitudes come up when optimizing.

        :param input_dim: dimensionality of the input space
        :param num_features: number of features to use; the higher the number of features, the better the approximation
                             of the SE kernel
        :param fourier_series_init: initializer used for the harmonics, i.e., the Fourier series; includes the
                                              amplitudes and phases of the harmonics
        :param length_scale: length scale to resemble; approximately equivalent to the length scale of the SE kernel
        :param device: device to run on
        """
        super().__init__(device=device)

        assert_positive(input_dim, "input_dim")
        assert_positive(num_features, "num_features")
        if length_scale is None:
            length_scale = torch.tensor(1.0, requires_grad=True)
        assert_positive(length_scale, "length_scale")  # Implies checking that n is a scalar.

        self._input_dim = input_dim
        self._num_features = num_features
        self._num_harmonics = fourier_series_init.num_harmonics
        self._length_scale = torch.nn.Parameter(length_scale, requires_grad=length_scale.requires_grad).to(device=self.device)
        self._amplitudes_sqrt = torch.nn.Parameter(data=fourier_series_init.amplitudes_sqrt, requires_grad=fourier_series_init.optimize_amplitudes).to(device=self.device)
        self._phases = torch.nn.Parameter(data=fourier_series_init.amplitudes_sqrt, requires_grad=fourier_series_init.optimize_phases).to(device=self.device)

        weight_distribution = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        bias_distribution = torch.distributions.Uniform(0.0, 2 * np.pi)
        self._weights = weight_distribution.sample((self._num_features,)).to(device=self.device)
        self._biases = bias_distribution.sample((self._num_features,)).to(device=self.device)

        self.register_parameters(self._length_scale, self._amplitudes_sqrt, self._phases)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the random Fourier series features."""

        assert_axis_length(x, 1, self._input_dim, "x")

        biases = self._biases.unsqueeze(dim=0).unsqueeze(dim=2)
        phases = self._phases.unsqueeze(dim=0).unsqueeze(dim=1)

        normalization = np.sqrt(2 / self._num_features)
        weighted_inputs = (x @ self._weights.T).unsqueeze(dim=2)
        harmonic_multipliers = (torch.arange(self._num_harmonics + 1, device=self.device)).unsqueeze(dim=0).unsqueeze(dim=1)
        harmonics_activations = harmonic_multipliers * weighted_inputs + biases + phases
        harmonics = self._amplitudes * torch.cos(harmonics_activations)
        # TODO: Is it helpful to normalize over the amplitudes?
        return normalization * harmonics.sum(dim=2)  # / self._amplitudes.sum()

    @property
    def _amplitudes(self) -> torch.Tensor:
        """Computes the amplitudes from the sqrt-amplitudes, i.e., squares them."""
        return self._amplitudes_sqrt ** 2
