import pytest
import torch
from sklearn.gaussian_process.kernels import DotProduct

from kernel.feature_based_kernel import FeatureBasedKernel
from kernel.sklearn_kernel_wrapper import SkLearnKernelWrapper


class MockFeatureKernel(FeatureBasedKernel):
    def forward_features(self, x: torch.Tensor, state=None) -> torch.Tensor:
        return x


@pytest.mark.parametrize("samples_p", [1, 2, 3, 5])
@pytest.mark.parametrize("samples_q", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_forward(samples_p, samples_q, dim):
    kernel = MockFeatureKernel()
    reference = SkLearnKernelWrapper(DotProduct())
    p = torch.rand(size=(samples_p, dim))
    q = torch.rand(size=(samples_q, dim))
    assert kernel(p, q).shape == reference(p, q).shape
