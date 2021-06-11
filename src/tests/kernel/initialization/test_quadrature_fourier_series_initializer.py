import warnings

import numpy as np
import pytest
import torch
from scipy.integrate import AccuracyWarning

from rfsf.kernel.initialization.quadrature_fourier_series_initializer import QuadratureFourierSeriesInitializer
from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer


@pytest.fixture(scope="function")
def disable_accuracy_warnings():
    warnings.simplefilter("ignore", category=AccuracyWarning)
    yield None
    warnings.simplefilter("always", category=AccuracyWarning)


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_constructor(num_harmonics, half_period, disable_accuracy_warnings):
    init = QuadratureFourierSeriesInitializer(QuadratureFourierSeriesInitializer.relu, num_harmonics, half_period, torch_dtype=torch.float)
    assert init.cosine_coefficients.shape == (num_harmonics + 1,)
    assert init.sine_coefficients.shape == (num_harmonics + 1,)
    assert init.amplitudes_sqrt.shape == (num_harmonics + 1,)
    assert init.phases.shape == (num_harmonics + 1,)
    assert torch.allclose(init.sine_coefficients[0], torch.tensor(0.0, dtype=torch.float))
    assert torch.allclose(init.amplitudes_sqrt[0] ** 2, init.cosine_coefficients[0])


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_coefficients_relu(num_harmonics, half_period, disable_accuracy_warnings):
    reference = ReLUFourierSeriesInitializer(num_harmonics, half_period)
    init = QuadratureFourierSeriesInitializer(QuadratureFourierSeriesInitializer.relu, num_harmonics, half_period, torch_dtype=torch.float)
    assert np.allclose(init.cosine_coefficients, reference.cosine_coefficients, atol=0.01)
    assert np.allclose(init.sine_coefficients, reference.sine_coefficients, atol=0.01)
    assert np.allclose(init.amplitudes_sqrt, reference.amplitudes_sqrt, atol=0.01)
    assert np.allclose(init.phases, reference.phases, atol=0.01)


@pytest.mark.parametrize("func", [QuadratureFourierSeriesInitializer.erf, QuadratureFourierSeriesInitializer.relu, QuadratureFourierSeriesInitializer.tanh])
@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_no_error(func, num_harmonics, half_period, disable_accuracy_warnings):
    init = QuadratureFourierSeriesInitializer(func, num_harmonics, half_period, torch_dtype=torch.float)
    init.compute_coefficients()
