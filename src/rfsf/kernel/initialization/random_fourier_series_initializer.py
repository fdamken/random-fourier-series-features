from typing import Tuple

import torch

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer


class RandomFourierSeriesInitializer(FourierSeriesInitializer):
    """
    Fourier series initialization where the cosine/sine coefficients are randomly drawn from a uniform distribution on
    the interval :math:`[0, 1]`.
    """

    def compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly draws the cosine/sine coefficients from a uniform distribution."""
        return tuple(torch.rand((2, self._num_harmonics + 1)))
