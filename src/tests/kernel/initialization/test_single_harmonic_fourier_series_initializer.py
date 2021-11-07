import numpy as np
import pytest
import torch
from pytest import approx

from rfsf.kernel.initialization.single_harmonic_fourier_series_initializer import SingleHarmonicFourierSeriesInitializer


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_constructor(num_harmonics, half_period):
    init = SingleHarmonicFourierSeriesInitializer(num_harmonics, half_period)
    amplitudes = torch.zeros((num_harmonics,))
    amplitudes[0] = torch.tensor(1.0)
    assert init.cosine_coefficients.shape == (num_harmonics,)
    assert init.sine_coefficients.shape == (num_harmonics,)
    assert init.amplitudes.shape == (num_harmonics,)
    assert init.phases.shape == (num_harmonics,)
    assert init.amplitudes == approx(amplitudes)
    assert init.phases == approx(torch.zeros((num_harmonics,)))


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_no_error(num_harmonics, half_period):
    init = SingleHarmonicFourierSeriesInitializer(num_harmonics, half_period)
    init.compute_coefficients()
