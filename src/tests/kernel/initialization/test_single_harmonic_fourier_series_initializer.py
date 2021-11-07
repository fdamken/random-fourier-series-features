import numpy as np
import pytest
import torch
from pytest import approx

from rfsf.kernel.initialization.single_harmonic_fourier_series_initializer import SingleHarmonicFourierSeriesInitializer


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_constructor(num_harmonics, half_period):
    init = SingleHarmonicFourierSeriesInitializer(num_harmonics, half_period)
    amplitudes = torch.zeros((num_harmonics + 1,))
    amplitudes[1] = torch.tensor(1.0)
    phases = torch.zeros((num_harmonics + 1,))
    phases[0] = np.pi / 3
    assert init.cosine_coefficients.shape == (num_harmonics + 1,)
    assert init.sine_coefficients.shape == (num_harmonics + 1,)
    assert init.amplitudes.shape == (num_harmonics + 1,)
    assert init.phases.shape == (num_harmonics + 1,)
    assert init.amplitudes == approx(amplitudes)
    assert init.phases == approx(phases)


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_no_error(num_harmonics, half_period):
    init = SingleHarmonicFourierSeriesInitializer(num_harmonics, half_period)
    init.compute_coefficients()
