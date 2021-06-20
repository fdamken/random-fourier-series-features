import gpytorch.settings
import pytest
import torch

from rfsf.kernel.initialization.random_fourier_series_initializer import RandomFourierSeriesInitializer
from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer
from rfsf.kernel.rfsf_kernel import RFSFKernel


def test_constructor():
    RFSFKernel(100, RandomFourierSeriesInitializer(10, 0))


def test_initialization():
    num_samples = 1
    half_period = 1
    init = ReLUFourierSeriesInitializer(8, half_period)
    rfsf = RFSFKernel(num_samples, init)
    rfsf.lengthscale = torch.tensor(1.0)
    x = torch.arange(-3 * half_period, 3 * half_period, 0.01)
    actual = rfsf._featurize(x.unsqueeze(-1), identity_randoms=True)[:, 0]
    expected = init(x, use_sine_cosine_form=False)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("batch_a", [1, 3])
@pytest.mark.parametrize("batch_b", [1, 3])
@pytest.mark.parametrize("samples_p", [1, 2, 3, 5])
@pytest.mark.parametrize("samples_q", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_compute(batch_a, batch_b, samples_p, samples_q, dim):
    rfsf = RFSFKernel(100, RandomFourierSeriesInitializer(10, 1))
    p = torch.rand(size=(batch_a, batch_b, samples_p, dim))
    q = torch.rand(size=(batch_a, batch_b, samples_q, dim))
    with gpytorch.settings.lazily_evaluate_kernels(False):
        cov = rfsf(p, q)
    assert cov.shape == (batch_a, batch_b, samples_p, samples_q)


@pytest.mark.parametrize("batch_a", [1, 3])
@pytest.mark.parametrize("batch_b", [1, 3])
@pytest.mark.parametrize("samples", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_compute_diag(batch_a, batch_b, samples, dim):
    rfsf = RFSFKernel(100, RandomFourierSeriesInitializer(10, 1))
    p = torch.rand(size=(batch_a, batch_b, samples, dim))
    q = torch.rand(size=(batch_a, batch_b, samples, dim))
    with gpytorch.settings.lazily_evaluate_kernels(False):
        cov = rfsf(p, q, diag=True)
    assert cov.shape == (batch_a, batch_b, samples)


@pytest.mark.parametrize("batch_a", [1, 3])
@pytest.mark.parametrize("batch_b", [1, 3])
@pytest.mark.parametrize("samples", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@pytest.mark.parametrize("num_samples", [1, 2, 3, 5])
def test_compute_eq(batch_a, batch_b, samples, dim, num_samples):
    rfsf = RFSFKernel(num_samples, RandomFourierSeriesInitializer(10, 1))
    p = torch.rand(size=(batch_a, batch_b, samples, dim))
    with gpytorch.settings.lazily_evaluate_kernels(False):
        cov = rfsf(p)
    assert cov.shape == (batch_a, batch_b, samples, samples)


@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@pytest.mark.parametrize("num_samples", [1, 2, 3, 5])
def test_random_weights_and_phases_shape(dim, num_samples):
    rfsf = RFSFKernel(num_samples, RandomFourierSeriesInitializer(10, 1))
    assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_WEIGHTS)
    assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_BIASES)
    weights_1, phases_1 = rfsf.get_weights_and_phases(dim)
    assert weights_1.shape == (dim, num_samples)
    assert phases_1.shape == (num_samples,)


@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@pytest.mark.parametrize("num_samples", [1, 2, 3, 5])
def test_random_weights_and_phases_buffer(dim, num_samples):
    rfsf = RFSFKernel(num_samples, RandomFourierSeriesInitializer(10, 1))
    assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_WEIGHTS)
    assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_BIASES)

    # Test if calling _get_weights_and_phases twice produces identical values.
    weights_1, phases_1 = rfsf.get_weights_and_phases(dim)
    weights_2, phases_2 = rfsf.get_weights_and_phases(dim)
    assert torch.allclose(weights_2, weights_1)
    assert torch.allclose(phases_2, phases_1)

    # Test if clearing produces new weights and phases.
    rfsf.clear_weights_and_phases()
    weights_3, phases_3 = rfsf.get_weights_and_phases(dim)
    assert not torch.allclose(weights_3, weights_2)
    assert not torch.allclose(phases_3, phases_2)
    weights_4, phases_4 = rfsf.get_weights_and_phases(dim)
    assert torch.allclose(weights_4, weights_3)
    assert torch.allclose(phases_4, phases_3)
