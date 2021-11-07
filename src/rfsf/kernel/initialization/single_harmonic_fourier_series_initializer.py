from typing import Tuple

import torch

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer


class SingleHarmonicFourierSeriesInitializer(FourierSeriesInitializer):
    """
    Initializes the coefficients of the Fourier series such that only the first harmonic is active, i.e., the other
    amplitudes are zero (or close to zero).
    """

    def _compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        amplitudes = torch.zeros((self.num_harmonics,))
        amplitudes[0] = torch.tensor(1.0)
        phases = torch.zeros((self.num_harmonics,))
        return amplitudes * phases.cos(), amplitudes * phases.sin()
