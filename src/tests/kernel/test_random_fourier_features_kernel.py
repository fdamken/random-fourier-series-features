import numpy as np
import pytest
import torch
from sklearn.gaussian_process.kernels import RBF

from rfsf.kernel.random_fourier_features_kernel import RandomFourierFeaturesKernel
from rfsf.kernel.sklearn_kernel_wrapper import SkLearnKernelWrapper


def test_constructor():
    RandomFourierFeaturesKernel(2, 100)


@pytest.mark.parametrize("samples_p", [1, 2, 3, 5])
@pytest.mark.parametrize("samples_q", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_compute(samples_p, samples_q, dim):
    rff = RandomFourierFeaturesKernel(dim, 100)
    p = torch.rand(size=(samples_p, dim))
    q = torch.rand(size=(samples_q, dim))
    rff(p, q)


@pytest.mark.parametrize("length_scale", [0.1, 1.0, 2.0])
def test_convergence(length_scale):
    rbf = SkLearnKernelWrapper(RBF(length_scale=length_scale))
    x = torch.arange(-5.0, 5.0, 0.01).unsqueeze(dim=1)
    expected = rbf(x, x)
    for num_features in 10 ** np.arange(10, dtype=int):
        rff = RandomFourierFeaturesKernel(1, num_features, length_scale=length_scale)
        actual = rff(x, x)
        if torch.allclose(actual, expected, atol=0.01):
            break
    else:
        # noinspection PyUnboundLocalVariable
        assert False, f"Random Fourier Features did not converge with at most {num_features} features!"  # pylint: disable=undefined-loop-variable
    print(f"Converged with {num_features} features!")  # pylint: disable=undefined-loop-variable
