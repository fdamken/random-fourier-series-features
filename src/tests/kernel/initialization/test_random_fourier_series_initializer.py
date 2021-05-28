import pytest

from rfsf.kernel.initialization.random_fourier_series_initializer import RandomFourierSeriesInitializer


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
def test_constructor(num_harmonics):
    init = RandomFourierSeriesInitializer(num_harmonics, 0.0)
    assert init.cosine_coefficients.shape == (num_harmonics + 1,)
    assert init.sine_coefficients.shape == (num_harmonics + 1,)
    assert init.amplitudes_sqrt.shape == (num_harmonics + 1,)
    assert init.phases.shape == (num_harmonics + 1,)
