import pytest
import torch

from rfsf.kernel.random_fourier_series_features_kernel import RandomFourierSeriesFeaturesKernel


def test_constructor():
    RandomFourierSeriesFeaturesKernel(2, 100, 10)


@pytest.mark.parametrize("samples_p", [1, 2, 3, 5])
@pytest.mark.parametrize("samples_q", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_compute(samples_p, samples_q, dim):
    rff = RandomFourierSeriesFeaturesKernel(dim, 100, 10)
    p = torch.rand(size=(samples_p, dim))
    q = torch.rand(size=(samples_q, dim))
    rff(p, q)


def test_parameters():
    rff = RandomFourierSeriesFeaturesKernel(1, 1, 1)
    assert len(rff.parameters) == 3
