from typing import Tuple

import torch

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer


class SingleHarmonicFourierSeriesInitializer(FourierSeriesInitializer):
    """
    Initializes the coefficients of the Fourier series such that only the first harmonic is active, i.e., the other
    amplitudes are zero (or close to zero).
    """

    @property
    def amplitudes(self) -> torch.Tensor:
        amplitudes = torch.ones((self.num_harmonics + 1,)) * 1e-5
        amplitudes[1] = torch.tensor(1.0)
        return amplitudes

    @property
    def phases(self) -> torch.Tensor:
        return torch.zeros((self.num_harmonics + 1,))

    def compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()  # TODO: This is currently a hacky implementation.
