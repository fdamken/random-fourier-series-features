import pytest
import torch
from sklearn.gaussian_process.kernels import DotProduct

from rfsf.kernel.feature_based_kernel import DegenerateKernel
from rfsf.kernel.sklearn_kernel_wrapper import SkLearnKernelWrapper


class MockFeatureKernel(DegenerateKernel):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
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
