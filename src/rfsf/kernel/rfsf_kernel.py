import math
from typing import NoReturn, Optional, Tuple

import numpy as np
import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyTensor, LowRankRootLazyTensor, MatmulLazyTensor, RootLazyTensor

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer
from rfsf.prediction_strategies.double_precision_prediction_strategy import DoublePrecisionPredictionStrategy
from rfsf.util.assertions import assert_positive


class RFSFKernel(Kernel):
    """
    Implementation of Random Fourier Series Features (RFSFs).

    .. warning::
        This class is part of ongoing research!
    """

    BUFFER_RAND_WEIGHTS = "rand_weights"
    BUFFER_RAND_BIASES = "rand_biases"

    has_lengthscale = True

    def __init__(self, num_samples: int, fourier_series_init: FourierSeriesInitializer, **kwargs):
        """
        Constructor.

        :param num_samples: number of samples, aka. features, to use
        :param fourier_series_init: initializer used for the harmonics, i.e., the Fourier series; includes the
                                    amplitudes and phases of the harmonics
        :param kwargs: other keyword arguments passed to :py:class:`gpytorch.kernels.Kernel`
        """
        super().__init__(**kwargs)

        assert_positive(num_samples, "num_samples")

        device = self.raw_lengthscale.device

        self.num_samples = num_samples
        self.num_harmonics = fourier_series_init.num_harmonics
        self.half_period = torch.nn.Parameter(torch.tensor(fourier_series_init.half_period), requires_grad=fourier_series_init.optimize_half_period).to(device=device)
        self.amplitudes_log = torch.nn.Parameter(data=fourier_series_init.amplitudes.log(), requires_grad=fourier_series_init.optimize_amplitudes).to(device=device)
        self.phases = torch.nn.Parameter(data=fourier_series_init.phases, requires_grad=fourier_series_init.optimize_phases).to(device=device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> LazyTensor:
        """
        Forwards the kernel computation.

        .. note::
            This code is copied from the RFF kernel included in GPyTorch.

        .. seealso::
            https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/kernels/rff_kernel.py
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self._featurize(x1)
        if x1_eq_x2:
            z2 = z1
        else:
            z2 = self._featurize(x2)
        D = float(self.num_samples)
        if diag:
            return (z1 * z2).sum(-1) / D
        if x1_eq_x2:
            # Exploit low rank structure, if there are fewer features than data points
            if z1.size(-1) < z2.size(-2):
                return LowRankRootLazyTensor(z1 / math.sqrt(D))
            return RootLazyTensor(z1 / math.sqrt(D))
        return MatmulLazyTensor(z1 / D, z2.transpose(-1, -2))

    def clear_weights_and_phases(self) -> NoReturn:
        """
        Clears the random weights and phases buffers (i.e., sets them to `None`) such that the weights and phases will
        be sampled again by :py:meth:`.get_weights_and_phases` (and indirectly by :py:meth:`.forward`).
        """
        self.register_buffer(RFSFKernel.BUFFER_RAND_WEIGHTS, None)
        self.register_buffer(RFSFKernel.BUFFER_RAND_BIASES, None)

    def get_weights_and_phases(self, num_dims: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the random weights and phases and stores them as buffers `rand_weights` and `rand_phases` if they do
        not exist yet. If they exist, this method does nothing. To clear and re-initialize the buffers, call
        :py:meth:`.clear_weights_and_phases`.

        :param num_dims: input dimensionality, `d`; not necessary if weights and phases are alread stored
        :return: stored or newly generated random weights; shape `(d, D)`
        :return: stored or newly generated random phases; shape `(D,)`
        """
        if hasattr(self, RFSFKernel.BUFFER_RAND_WEIGHTS) and hasattr(self, RFSFKernel.BUFFER_RAND_BIASES):
            rand_weights, rand_phases = getattr(self, RFSFKernel.BUFFER_RAND_WEIGHTS), getattr(self, RFSFKernel.BUFFER_RAND_BIASES)
        else:
            rand_weights, rand_phases = None, None
        if rand_weights is None or rand_phases is None:
            assert num_dims is not None, "num_dims must not be None if buffers are empty"
            rand_weights = torch.randn((num_dims, self.num_samples), dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
            rand_phases = 2 * np.pi * torch.rand((self.num_samples,), dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
            rand_weights, rand_phases = self.set_weights_and_phases(rand_weights, rand_phases)
        return rand_weights, rand_phases

    def set_weights_and_phases(self, rand_weights: torch.Tensor, rand_phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sets the random weights and phases buffers to the given weights and phases. This can be used for getting a
        deterministic kernel output, e.g., when animating the mean over multiple iterations.

        :param rand_weights: random weights to set; shape `(d, D)`
        :param rand_phases: random phases to set; shape `(D,)`
        :return: random weights; shape `(d, D)`
        :return: random phases; shape `(D,)`
        """
        self.register_buffer(RFSFKernel.BUFFER_RAND_WEIGHTS, rand_weights)
        self.register_buffer(RFSFKernel.BUFFER_RAND_BIASES, rand_phases)
        return rand_weights, rand_phases

    def _featurize(self, x: torch.Tensor, *, identity_randoms: bool = False) -> torch.Tensor:
        """
        Calculates the RFSF features that are then used for computing the kernel.

        :param x: input data; shape `(..., n, d)`
        :return: feature vector; shape `(..., n, D)`
        """

        rand_weights, rand_phases = self.get_weights_and_phases(x.size(-1))
        amplitudes = self._amplitudes
        phases = self.phases

        if identity_randoms:
            rand_weights = torch.ones_like(rand_weights)
            rand_phases = torch.zeros_like(rand_phases)

        n = torch.arange(0, self.num_harmonics + 1, dtype=x.dtype, device=self.raw_lengthscale.device)
        weighted_inputs = x.matmul(rand_weights / self.lengthscale.transpose(-1, -2)).unsqueeze(dim=-1)
        harmonized_inputs = np.pi / self.half_period * weighted_inputs @ n.unsqueeze(dim=0)
        del n, weighted_inputs  # Not needed anymore --> save memory.
        harmonics_activations = harmonized_inputs - phases + rand_phases.unsqueeze(dim=-1)
        del harmonized_inputs  # Not needed anymore --> save memory.
        harmonics = amplitudes * torch.cos(harmonics_activations)
        del harmonics_activations  # Not needed anymore --> save memory.

        return harmonics.sum(dim=-1) / amplitudes.sum()

    @property
    def _amplitudes(self) -> torch.Tensor:
        """Computes the amplitudes from the log-amplitudes."""
        return self.amplitudes_log.exp()

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return DoublePrecisionPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)
