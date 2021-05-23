from typing import Tuple

import numpy as np
import torch

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer


class ReLUFourierSeriesInitializer(FourierSeriesInitializer):
    """
    Initializes the coefficients of the Fourier series such that they resemble a Rectified Linear Unit (ReLU).
    """

    def __init__(self, num_harmonics: int, half_period: float, optimize_amplitudes: bool = True, optimize_phases: bool = True):
        """
        Constructor.

        :param num_harmonics: number of harmonics to use; generally, the more harmonics are used, the better the
                              initialization function can be approximated; has to be positive
        :param half_period: half the assumed period after which the ReLU repeats
        :param optimize_amplitudes: whether to optimize the amplitudes later on; causes `required_grad` of the
                                    corresponding parameter in the RFSF kernel to be set to `True`
        :param optimize_phases: whether to optimize the phases later on; causes `required_grad` of the corresponding
                                parameter in the RFSF kernel to be set to `True`
        """
        super().__init__(num_harmonics, optimize_amplitudes, optimize_phases)
        self._half_period = half_period

    def compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the cosine/sine coefficients to resemble a ReLU."""

        a = torch.zeros((self.num_harmonics + 1,))
        b = torch.zeros((self.num_harmonics + 1,))

        k = torch.arange(1, self.num_harmonics + 1, dtype=torch.int)
        a[0] = self._half_period / 2
        a[1:] = (self._half_period * (-1) ** k - self._half_period) / (k ** 2 * np.pi ** 2)
        b[1:] = -(self._half_period * (-1) ** k) / (k * np.pi)

        return a, b
