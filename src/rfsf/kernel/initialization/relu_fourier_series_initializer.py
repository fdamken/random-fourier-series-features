from typing import Tuple

import numpy as np
import torch

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer


class ReLUFourierSeriesInitializer(FourierSeriesInitializer):
    """
    Initializes the coefficients of the Fourier series such that they resemble a Rectified Linear Unit (ReLU).
    """

    def compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the cosine/sine coefficients to resemble a ReLU."""

        a = torch.zeros((self.num_harmonics + 1,))
        b = torch.zeros((self.num_harmonics + 1,))

        k = torch.arange(1, self.num_harmonics + 1, dtype=torch.int)
        a[0] = self.half_period / 2
        a[1:] = (self.half_period * (-1) ** k - self.half_period) / (k ** 2 * np.pi ** 2)
        b[1:] = -(self.half_period * (-1) ** k) / (k * np.pi)

        return a, b
