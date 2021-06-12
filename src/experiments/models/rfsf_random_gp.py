import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

from rfsf.kernel.initialization.random_fourier_series_initializer import RandomFourierSeriesInitializer
from rfsf.kernel.rfsf_kernel import RFSFKernel


class RFSFRandomGP(ExactGP):
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood: Likelihood,
        num_samples: int,
        num_harmonics: int,
        half_period: float,
        optimize_amplitudes: bool,
        optimize_phases: bool,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.cov_module = RFSFKernel(num_samples, RandomFourierSeriesInitializer(num_harmonics, half_period, optimize_amplitudes, optimize_phases))

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return MultivariateNormal(mean, cov)