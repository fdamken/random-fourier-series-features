import math
from typing import NoReturn

import numpy as np
import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyTensor, LowRankRootLazyTensor, MatmulLazyTensor, RootLazyTensor

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer
from rfsf.util.assertions import assert_positive


class RandomFourierSeriesFeaturesKernel(Kernel):
    """
    Implementation of Random Fourier Series Features (RFSFs).

    .. warning::
        This class is part of ongoing research!
    """

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

        self.num_samples = num_samples
        self.num_harmonics = fourier_series_init.num_harmonics
        self._amplitudes_sqrt = torch.nn.Parameter(data=fourier_series_init.amplitudes_sqrt, requires_grad=fourier_series_init.optimize_amplitudes).to(
            device=self.raw_lengthscale.device)
        self._phases = torch.nn.Parameter(data=fourier_series_init.amplitudes_sqrt, requires_grad=fourier_series_init.optimize_phases).to(device=self.raw_lengthscale.device)

        self.register_parameter("amplitudes_sqrt", self._amplitudes_sqrt)
        self.register_parameter("phases", self._phases)

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
        num_dims = x1.size(-1)
        if not hasattr(self, "randn_weights") or not hasattr(self, "rand_phases"):
            self._init_weights_and_phases(num_dims, self.num_samples)
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

    def _init_weights_and_phases(self, num_dims: int, num_samples: int) -> NoReturn:
        """Initializes the random weights and phases and stores them as buffers `rand_weights` and `rand_phases`."""
        rand_weights = torch.randn((num_dims, num_samples), dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        rand_phases = 2 * np.pi * torch.rand((num_samples,), dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        self.register_buffer("rand_weights", rand_weights)
        self.register_buffer("rand_phases", rand_phases)

    def _featurize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the RFSF features that are then used for computing the kernel.

        :param x: input data; shape `(..., n, d)`
        :return: feature vector; shape `(..., n, D)`
        """
        rand_weights = self.rand_weights
        rand_phases = self.rand_phases
        amplitudes = self._amplitudes
        phases = self._phases

        n = torch.arange(self.num_harmonics + 1, dtype=x.dtype, device=self.raw_lengthscale.device)
        weighted_inputs = x.matmul(rand_weights / self.lengthscale.transpose(-1, -2)).unsqueeze(dim=-1)
        harmonized_inputs = weighted_inputs @ n.unsqueeze(dim=0)
        harmonics_activations = harmonized_inputs + phases + rand_phases.unsqueeze(dim=-1)
        harmonics = amplitudes * torch.cos(harmonics_activations)

        # TODO: Is it helpful to normalize over the amplitudes?
        return 2 * harmonics.sum(dim=-1)

    @property
    def _amplitudes(self) -> torch.Tensor:
        """Computes the amplitudes from the sqrt-amplitudes, i.e., squares them."""
        return self._amplitudes_sqrt ** 2
