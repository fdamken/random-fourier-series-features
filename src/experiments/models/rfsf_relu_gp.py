import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer
from rfsf.kernel.rfsf_kernel import RFSFKernel


class RFSFReLUGP(ExactGP):
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood: Likelihood,
        *,
        num_samples: int,
        num_harmonics: int,
        half_period: float,
        optimize_amplitudes: bool,
        optimize_phases: bool,
        optimize_half_period: bool,
        use_sine_cosine_form: bool,
        use_ard: bool,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.mean_module.constant.requires_grad = False
        self.cov_module = RFSFKernel(
            num_samples,
            ReLUFourierSeriesInitializer(num_harmonics, half_period, optimize_amplitudes, optimize_phases, optimize_half_period),
            use_sine_cosine_form=use_sine_cosine_form,
            ard_num_dims=train_inputs.shape[1] if use_ard else None,
        )

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return MultivariateNormal(mean, cov)
