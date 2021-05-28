from typing import Callable, Optional, Tuple

import numpy as np
import scipy as sp
import scipy.special
import torch
from scipy.integrate import quadrature

from rfsf.kernel.initialization.fourier_series_initializer import FourierSeriesInitializer
from rfsf.util.assertions import assert_positive


class QuadratureFourierSeriesInitializer(FourierSeriesInitializer):
    """
    General-purpose initializer to resemble any function, computing the coefficients of the Fourier series with
    Gaussian quadrature (`scipy.integrate.quadrature`).
    """

    @staticmethod
    def erf(x: np.ndarray) -> np.ndarray:
        return sp.special.erf(x)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(np.zeros_like(x), x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        num_harmonics: int,
        half_period: float,
        optimize_amplitudes: bool = True,
        optimize_phases: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        quadrature_maxiter: int = 200,
        quadrature_kwargs: dict = None,
    ):
        """
        Constructor.

        :param func: function to integrate; takes a batch of scalar values and produces a same-sized batch of output
                     values; is called with a modulus operator around, to periodicity according to the given period is
                     assured; hence, all interesting this have to happen around zero
        :param num_harmonics: number of harmonics to use; generally, the more harmonics are used, the better the
                              initialization function can be approximated; has to be positive
        :param half_period: half the assumed period after which the ReLU repeats
        :param optimize_amplitudes: whether to optimize the amplitudes later on; causes `required_grad` of the
                                    corresponding parameter in the RFSF kernel to be set to `True`
        :param optimize_phases: whether to optimize the phases later on; causes `required_grad` of the corresponding
                                parameter in the RFSF kernel to be set to `True`
        :param torch_dtype: torch data type to use for the coefficients; defaults to the torch default
        :param quadrature_maxiter: maximum order of Gaussian quadrature; defaults to `200` which should be enough for
                                   any reasonable amount of harmonics (:math:`< 500`); use higher values for more
                                   harmonics
        :param quadrature_kwargs: keyword arguments to pass to the `scipy.integration.quadrature` call; all arguments
                                  except for `func`, `a`, `b`, and `maxiter` are allowed; the maximum oder of Gaussian
                                  quadrature of can be set with the `quadrature_maxiter` parameter
        :param quadrature_kwargs:
        """
        super().__init__(num_harmonics, half_period, optimize_amplitudes, optimize_phases)
        if quadrature_kwargs is None:
            quadrature_kwargs = {}
        assert_positive(quadrature_maxiter, "quadrature_maxiter")
        self._func = func
        self._torch_dtype = torch_dtype
        self._quadrature_maxiter = quadrature_maxiter
        self._quadrature_kwargs = quadrature_kwargs

    def compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the cosine/sine coefficients using Gaussian quadrature."""

        func = lambda x: self._func((x + self.half_period) % (2 * self.half_period) - self.half_period)
        func_cosine_coefficients = lambda k: lambda x: func(x) * np.cos(np.pi / self.half_period * k * x)
        func_sine_coefficients = lambda k: lambda x: func(x) * np.sin(np.pi / self.half_period * k * x)

        interval_lo = -self.half_period
        interval_up = +self.half_period

        quadrature_kwargs = {"a": interval_lo, "b": interval_up, "maxiter": self._quadrature_maxiter}
        quadrature_kwargs.update(self._quadrature_kwargs)
        cosine_coefficients = [(quadrature(func, **quadrature_kwargs))[0] / self.half_period]
        sine_coefficients = [0.0]
        for n in range(1, self.num_harmonics + 1):
            cosine_coefficients.append(quadrature(func_cosine_coefficients(n), **quadrature_kwargs)[0] / self.half_period)
            sine_coefficients.append(quadrature(func_sine_coefficients(n), **quadrature_kwargs)[0] / self.half_period)

        return torch.tensor(cosine_coefficients, dtype=self._torch_dtype), torch.tensor(sine_coefficients, dtype=self._torch_dtype)
