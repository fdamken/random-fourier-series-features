import numpy as np
import pytest
import torch

from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer


@pytest.mark.parametrize("num_harmonics", [1, 2, 3, 5, 7, 11])
@pytest.mark.parametrize("half_period", [np.pi, 2 * np.pi, 10])
def test_constructor(num_harmonics, half_period):
    init = ReLUFourierSeriesInitializer(num_harmonics, half_period)
    assert init.cosine_coefficients.shape == (num_harmonics,)
    assert init.sine_coefficients.shape == (num_harmonics,)
    assert init.amplitudes.shape == (num_harmonics,)
    assert init.phases.shape == (num_harmonics,)


def test_convergence():
    half_period = np.pi
    x = torch.arange(-1.5 * half_period, 1.5 * half_period, 0.01)
    expected = torch.maximum(torch.zeros_like(x), (x + half_period) % (2 * half_period) - half_period)
    expected -= half_period / 4  # Normalize because the init-series expects functions to have mean zero.
    for num_harmonics in 10 ** np.arange(5, dtype=int):
        init = ReLUFourierSeriesInitializer(num_harmonics, half_period)
        actual = init(x)
        if torch.linalg.norm(actual - expected) < 0.1:
            break
    else:
        # noinspection PyUnboundLocalVariable
        assert False, f"ReLU initialization did not converge with at most {num_harmonics} harmonics!"  # pylint: disable=undefined-loop-variable
    print(f"Converged with {num_harmonics} harmonics!")  # pylint: disable=undefined-loop-variable
