import numpy as np
import pytest
import torch

from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_constructor(num_harmonics, half_period):
    init = ReLUFourierSeriesInitializer(num_harmonics, half_period)
    assert init.cosine_coefficients.shape == (num_harmonics + 1,)
    assert init.sine_coefficients.shape == (num_harmonics + 1,)
    assert init.amplitudes_sqrt.shape == (num_harmonics + 1,)
    assert init.phases.shape == (num_harmonics + 1,)
    assert torch.isclose(init.sine_coefficients[0], torch.tensor(0.0))
    assert torch.isclose(init.amplitudes_sqrt[0] ** 2, init.cosine_coefficients[0])
