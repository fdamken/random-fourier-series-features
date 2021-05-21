import numpy as np
import pytest
import torch
from sklearn.gaussian_process.kernels import RBF

from kernel.random_fourier_feature_kernel import RandomFourierFeatureKernel
from kernel.sklearn_kernel_wrapper import SkLearnKernelWrapper


def test_constructor():
    RandomFourierFeatureKernel(2, 100)


@pytest.mark.parametrize("samples_p", [1, 2, 3, 5])
@pytest.mark.parametrize("samples_q", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_compute(samples_p, samples_q, dim):
    rff = RandomFourierFeatureKernel(dim, 100)
    p = torch.rand(size=(samples_p, dim))
    q = torch.rand(size=(samples_q, dim))
    rff(p, q)


def test_convergence():
    length_scale = 1.0
    rbf = SkLearnKernelWrapper(RBF(length_scale=length_scale))
    x = torch.arange(-5.0, 5.0, 0.01).unsqueeze(dim=1)
    expected = rbf(x, x)
    for num_features in 10 ** np.arange(0, 10 + 1, dtype=int):
        rff = RandomFourierFeatureKernel(1, num_features, length_scale=length_scale)
        actual = rff(x, x)
        if torch.allclose(actual, expected, atol=0.01):
            break
    else:
        # noinspection PyUnboundLocalVariable
        assert False, f"Random Fourier Features did not converge with at most {num_features} features!"  # pylint: disable=undefined-loop-variable
    print(f"Converged with {num_features} features!")  # pylint: disable=undefined-loop-variable
