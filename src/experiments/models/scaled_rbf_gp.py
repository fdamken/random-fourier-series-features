import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class ScaledRBFGP(ExactGP):
    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.cov_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.cov_module(x)
        return MultivariateNormal(mean, cov)
