from typing import Tuple

import numpy as np
import torch

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer


class NonNegativeSineInitializer(FourierSeriesInitializer):
    def __init__(self):
        super().__init__(1, np.pi)

    def _compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Represents a pure sine wave that with an amplitude of nine, i.e., f(x) = 9 sin(x).
        return torch.tensor([0.0]), torch.tensor([9.0])


def test_non_negative_sine_coefficients():
    init = NonNegativeSineInitializer()
    assert torch.allclose(init.cosine_coefficients, torch.tensor([0.0]))
    assert torch.allclose(init.sine_coefficients, torch.tensor([9.0]))
    assert torch.allclose(init.amplitudes, torch.tensor([9.0]))
    assert torch.allclose(init.phases, torch.tensor([np.pi / 2]))


# noinspection PyTypeChecker
def test_non_negative_sine_call():
    init = NonNegativeSineInitializer()
    x = torch.arange(-10, 10, 0.1)
    expected = 9 * torch.sin(x)
    actual_sine_cosine = init(x, use_sine_cosine_form=True)
    actual_amplitude_phase = init(x, use_sine_cosine_form=False)
    assert torch.allclose(actual_sine_cosine, expected)
    assert torch.allclose(actual_amplitude_phase, expected, atol=1e-5)
    assert torch.allclose(actual_amplitude_phase, actual_sine_cosine, atol=1e-5)
