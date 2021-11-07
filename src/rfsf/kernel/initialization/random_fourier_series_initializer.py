from typing import Tuple

import torch

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer


class RandomFourierSeriesInitializer(FourierSeriesInitializer):
    """
    Initializes the coefficients of the Fourier series randomly, drawn from a uniform distribution on :math:`[0, 1]`.
    """

    def _compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the cosine/sine coefficients from a uniform distribution."""
        return tuple(torch.rand((2, self._num_harmonics)))
