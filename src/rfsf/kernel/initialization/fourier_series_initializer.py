from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch

from rfsf.util.assertions import assert_positive, assert_shape


class FourierSeriesInitializer(ABC):
    """
    Helper class for initializing the Fourier series of random Fourier series features. It implements code for computing
    the amplitudes and phases from cosine/sine coefficients which can simply be obtained by integration. Subclasses can
    implement various different initialization techniques, e.g., random initialization or actually using the Fourier
    series of any function.

    Note that this class does not represent fourier series at their full glory as the bias is ignored. Hence only
    functions with with mean zero can be represented accurately. This simplifies subsequent learning and reduces some
    ambiguity.
    """

    def __init__(self, num_harmonics: int, half_period: float, optimize_amplitudes: bool = True, optimize_phases: bool = True, optimize_half_period: bool = True):
        """
        Constructor.

        :param num_harmonics: number of harmonics to use; generally, the more harmonics are used, the better the
                              initialization function can be approximated; has to be positive
        :param half_period: initial value of half the assumed period after which the function repeats
        :param optimize_amplitudes: whether to optimize the amplitudes later on; causes `requires_grad` of the
                                    corresponding parameter in the RFSF kernel to be set to `True`
        :param optimize_phases: whether to optimize the phases later on; causes `requires_grad` of the corresponding
                                parameter in the RFSF kernel to be set to `True`
        :param optimize_half_period: whether to optimize the half-period value; causes `requires_grad` of the
                                     corresponding parameter in the RFSF kernel to be set to `True`.
        """
        assert_positive(num_harmonics, "num_harmonics")

        self._num_harmonics = num_harmonics
        self._half_period = half_period
        self._optimize_amplitudes = optimize_amplitudes
        self._optimize_phases = optimize_phases
        self._optimize_half_period = optimize_half_period

        self._cosine_coefficients = None
        self._sine_coefficients = None
        self._amplitudes: Optional[torch.Tensor] = None
        self._phases: Optional[torch.Tensor] = None

    def __call__(self, x: torch.Tensor, *, use_sine_cosine_form: bool = True) -> torch.Tensor:
        x = x.unsqueeze(dim=1)
        if use_sine_cosine_form:
            a = self.cosine_coefficients
            b = self.sine_coefficients
            n = torch.arange(1, self.num_harmonics + 1).unsqueeze(dim=0)
            cos = torch.cos(np.pi / self.half_period * n * x)
            sin = torch.sin(np.pi / self.half_period * n * x)
            result = (a.unsqueeze(dim=0) * cos + b.unsqueeze(dim=0) * sin).sum(dim=1)
        else:
            amplitudes = self.amplitudes.unsqueeze(dim=0)
            phases = self.phases.unsqueeze(dim=0)
            n = torch.arange(1, self.num_harmonics + 1).unsqueeze(dim=0)
            harmonics_activations = np.pi / self.half_period * n * x - phases
            harmonics = amplitudes * torch.cos(harmonics_activations)
            result = harmonics.sum(dim=1)
        return result

    @abstractmethod
    def _compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the cosine/sine coefficients.

        :return: tuple `(cosine_coefficients, sine_coefficients)`; the cosine/sine coefficients; each has shape
                 `(num_harmonics,)`, where the first entry of the sine coefficients is zero for real fourier series
        """
        raise NotImplementedError()  # pragma: no cover

    def compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the cosine/sine coefficients and checks their shapes.

        :return: tuple `(cosine_coefficients, sine_coefficients)`; the cosine/sine coefficients; each has shape
                 `(num_harmonics,)`, where the first entry of the sine coefficients is zero for real fourier series
        """
        cosine_coefficients, sine_coefficients = self._compute_coefficients()
        assert_shape(cosine_coefficients, (self._num_harmonics,), "cosine_coefficients")
        assert_shape(sine_coefficients, (self._num_harmonics,), "sine_coefficients")
        return cosine_coefficients, sine_coefficients

    @property
    def cosine_coefficients(self) -> torch.Tensor:
        """
        Gets (and lazily computes) the cosine coefficients.

        :return: cosine coefficients; shape `(num_harmonics,)`
        """
        if self._amplitudes is None:
            self._cosine_coefficients, self._sine_coefficients = self._compute_coefficients()
        return self._cosine_coefficients

    @property
    def sine_coefficients(self) -> torch.Tensor:
        """
        Gets (and lazily computes) the sine coefficients.

        :return: sine coefficients; shape `(num_harmonics,)`
        """
        if self._amplitudes is None:
            self._cosine_coefficients, self._sine_coefficients = self._compute_coefficients()
        return self._sine_coefficients

    @property
    def amplitudes(self) -> torch.Tensor:
        """
        Gets (and lazily computes) the amplitudes.

        :return: amplitudes; shape `(num_harmonics,)`
        """
        if self._amplitudes is None:
            self._amplitudes = self._compute_amplitudes()
        return self._amplitudes

    @property
    def phases(self) -> torch.Tensor:
        """
        Gets (and lazily computes) the phases. The first entry is always :math:`\\pi/3` to pull the first amplitude into
        the sum.

        :return: phases; shape `(num_harmonics,)`
        """
        if self._phases is None:
            self._phases = self._compute_phases()
        return self._phases

    @property
    def num_harmonics(self) -> int:
        """Gets the number of harmonics."""
        return self._num_harmonics

    @property
    def half_period(self) -> float:
        """Gets the half period."""
        return self._half_period

    @property
    def optimize_amplitudes(self) -> bool:
        """Gets whether to optimize the amplitudes."""
        return self._optimize_amplitudes

    @property
    def optimize_phases(self) -> bool:
        """Gets whether to optimize the phases."""
        return self._optimize_phases

    @property
    def optimize_half_period(self) -> bool:
        """Gets whether to optimize the half-period."""
        return self._optimize_half_period

    def _compute_amplitudes(self) -> torch.Tensor:
        """
        Computes the amplitudes from the cosine/sine coefficients. The amplitudes are computed using
        :math:`A_k = \\sqrt{a_k^2 + b_k^2}`, where :math:`a_k` and :math:`b_k` are the cosine/sine coefficients,
        respectively.

        :return: amplitudes; shape `(num_harmonics,)`
        """
        return torch.sqrt(self.cosine_coefficients ** 2 + self.sine_coefficients ** 2)

    def _compute_phases(self) -> torch.Tensor:
        """
        Computes the phases from the cosine/sine coefficients using :math:`\\mathrm{atan2}(b_k, a_k)`, where :math:`a_k`
        and :math:`b_k` are the cosine/sine coefficients, respectively.

        .. note::
            Method :py:meth:`.compute_coefficients` has to be executed beforehand, which is usually done in the
            constructor.

        :return: phases; shape `(num_harmonics,)`
        """
        return torch.atan2(self.sine_coefficients, self.cosine_coefficients)
