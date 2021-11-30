import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class RBFGP(ExactGP):
    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor, likelihood: Likelihood, *, use_ard: bool):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean().requires_grad_(False)
        self.cov_module = RBFKernel(ard_num_dims=train_inputs.shape[1] if use_ard else None)

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return MultivariateNormal(mean, cov)
