import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RFFKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class RFFGP(ExactGP):
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood: Likelihood,
        *,
        num_samples: int,
        use_ard: bool,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.mean_module.constant.requires_grad = False
        self.cov_module = RFFKernel(
            num_samples,
            ard_num_dims=train_inputs.shape[1] if use_ard else None,
        )

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return MultivariateNormal(mean, cov)
