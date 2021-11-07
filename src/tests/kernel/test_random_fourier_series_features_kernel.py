import gpytorch.settings
import pytest
import torch

from rfsf.kernel.initialization.random_fourier_series_initializer import RandomFourierSeriesInitializer
from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer
from rfsf.kernel.rfsf_kernel import RFSFKernel


def test_constructor():
    RFSFKernel(100, RandomFourierSeriesInitializer(10, 0.0))


@pytest.mark.parametrize("use_sine_cosine_form", [True, False])
def test_initialization(use_sine_cosine_form):
    num_samples = 1
    half_period = 1.0
    init = ReLUFourierSeriesInitializer(8, half_period)
    rfsf = RFSFKernel(num_samples, init, use_sine_cosine_form=use_sine_cosine_form)
    rfsf.lengthscale = torch.tensor(1.0)
    x = torch.arange(-3 * half_period, 3 * half_period, 0.01)
    actual_features = rfsf._featurize(x.unsqueeze(-1), identity_randoms=True)
    if use_sine_cosine_form:
        # The feature form separates cosine/sine into different features, so sum them up to compare to Fourier series.
        actual = actual_features[:, 0] + actual_features[:, 1]
    else:
        actual = actual_features[:, 0]
    expected = init(x, use_sine_cosine_form=use_sine_cosine_form)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("use_sine_cosine_form", [True, False])
@pytest.mark.parametrize("batch_a", [1, 3])
@pytest.mark.parametrize("batch_b", [1, 3])
@pytest.mark.parametrize("samples_p", [1, 2, 3, 5])
@pytest.mark.parametrize("samples_q", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_compute(use_sine_cosine_form, batch_a, batch_b, samples_p, samples_q, dim):
    rfsf = RFSFKernel(100, RandomFourierSeriesInitializer(10, 1.0), use_sine_cosine_form=use_sine_cosine_form)
    p = torch.rand(size=(batch_a, batch_b, samples_p, dim))
    q = torch.rand(size=(batch_a, batch_b, samples_q, dim))
    with gpytorch.settings.lazily_evaluate_kernels(False):
        cov = rfsf(p, q)
    assert cov.shape == (batch_a, batch_b, samples_p, samples_q)


@pytest.mark.parametrize("use_sine_cosine_form", [True, False])
@pytest.mark.parametrize("batch_a", [1, 3])
@pytest.mark.parametrize("batch_b", [1, 3])
@pytest.mark.parametrize("samples", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_compute_diag(use_sine_cosine_form, batch_a, batch_b, samples, dim):
    rfsf = RFSFKernel(100, RandomFourierSeriesInitializer(10, 1.0), use_sine_cosine_form=use_sine_cosine_form)
    p = torch.rand(size=(batch_a, batch_b, samples, dim))
    q = torch.rand(size=(batch_a, batch_b, samples, dim))
    with gpytorch.settings.lazily_evaluate_kernels(False):
        cov = rfsf(p, q, diag=True)
    assert cov.shape == (batch_a, batch_b, samples)


@pytest.mark.parametrize("use_sine_cosine_form", [True, False])
@pytest.mark.parametrize("batch_a", [1, 3])
@pytest.mark.parametrize("batch_b", [1, 3])
@pytest.mark.parametrize("samples", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@pytest.mark.parametrize("num_samples", [1, 2, 3, 5])
def test_compute_eq(use_sine_cosine_form, batch_a, batch_b, samples, dim, num_samples):
    rfsf = RFSFKernel(num_samples, RandomFourierSeriesInitializer(10, 1.0), use_sine_cosine_form=use_sine_cosine_form)
    p = torch.rand(size=(batch_a, batch_b, samples, dim))
    with gpytorch.settings.lazily_evaluate_kernels(False):
        cov = rfsf(p)
    assert cov.shape == (batch_a, batch_b, samples, samples)


@pytest.mark.parametrize("use_sine_cosine_form", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@pytest.mark.parametrize("num_samples", [1, 2, 3, 5])
def test_random_components_shape(use_sine_cosine_form, dim, num_samples):
    rfsf = RFSFKernel(num_samples, RandomFourierSeriesInitializer(10, 1.0), use_sine_cosine_form=use_sine_cosine_form)
    assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_WEIGHTS)
    assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_BIASES)
    weights_1, phases_1 = rfsf.get_random_components(dim)
    assert weights_1.shape == (dim, num_samples)
    if not use_sine_cosine_form:
        assert phases_1.shape == (num_samples,)


@pytest.mark.parametrize("use_sine_cosine_form", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@pytest.mark.parametrize("num_samples", [1, 2, 3, 5])
def test_random_components_buffer(use_sine_cosine_form, dim, num_samples):
    rfsf = RFSFKernel(num_samples, RandomFourierSeriesInitializer(10, 1.0), use_sine_cosine_form=use_sine_cosine_form)
    assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_WEIGHTS)
    if not use_sine_cosine_form:
        assert not hasattr(rfsf, RFSFKernel.BUFFER_RAND_BIASES)

    # Test if calling _get_weights_and_phases twice produces identical values.
    weights_1, phases_1 = rfsf.get_random_components(dim)
    weights_2, phases_2 = rfsf.get_random_components(dim)
    assert torch.allclose(weights_2, weights_1)
    if not use_sine_cosine_form:
        assert torch.allclose(phases_2, phases_1)

    # Test if clearing produces new weights and phases.
    rfsf.clear_random_components()
    weights_3, phases_3 = rfsf.get_random_components(dim)
    assert not torch.allclose(weights_3, weights_2)
    if not use_sine_cosine_form:
        assert not torch.allclose(phases_3, phases_2)
    weights_4, phases_4 = rfsf.get_random_components(dim)
    assert torch.allclose(weights_4, weights_3)
    if not use_sine_cosine_form:
        assert torch.allclose(phases_4, phases_3)
