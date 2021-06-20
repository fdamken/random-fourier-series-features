from typing import Callable, Optional

import numpy as np
import progressbar


class NumberTrendWidget(progressbar.Widget):
    """
    A progressbar widget that indicates the current trend of an observable value by displaying a right-, up-, or down-
    pointing triangle (`▶`, `▲`, or `▼`) besides the value. The observable can for example be the loss of a learning
    process or a likelihood. The trend is then depending on the last value the observable had when invoking
    :py:meth:`progressbar.ProgressBar.update`, where a right-, up-, and down-pointing triangle represent a non-changing
    (sideways), increasing, and decreasing trend, respectively. Equality is determined using :py:meth:`np.isclose`, and
    different `rtol` and `atol` parameters can be passed to the constructor.

    If the new value is neither equal to (w.r.t. :py:meth:`np.isclose`), less than, nor greater than the previous value,
    a warning sign (`⚠`) is printed. This should never happen!

    The widget takes a formatting string that the observable value is applied using the :py:meth:`format` method, and
    afterwards the indicator is appended to the resulting string. The observable function is allowed to return `None`,
    causing a placeholder to be shown. This placeholder is a string with as many spaces as the formatting would take for
    displaying the number `0.0`. In the first step (where there is no previous value), two empty spaces are appended to
    the formatted string.
    """

    def __init__(self, format_str: str, observable: Callable[[], Optional[float]], rtol: Optional[float] = 1e-5, atol: Optional[float] = 1e-8):
        """
        Constructor.

        :param format_str: formatting string to use; see class documentation for more information
        :param observable: callable that returns the current value of the observable; used in the :py:meth:`.update`
                           function for determining the trend indicator
        :param rtol: relative tolerance used for checking for equality; see :py:meth:`np.isclose` for more information;
                     defaults to `1e-5`, NumPy's default
        :param atol: absolute tolerance used for checking for equality; see :py:meth:`np.isclose` for more information;
                     defaults to `1e-8`, NumPy's default
        """
        self._format = format_str
        self._observable = observable
        self._rtol = rtol
        self._atol = atol
        self._placeholder = " " * (len(format(0.0, self._format)) + 2)
        self._previous = None

    def update(self, _):
        value = self._observable()
        self._previous = value
        if value is None:
            return self._placeholder
        if self._previous is not None:
            if np.isclose(value, self._previous, rtol=self._rtol, atol=self._atol):
                suffix = " \u25B6"  # Black right-pointing triangle.
            elif value > self._previous:
                suffix = " \u25B2"  # Black up-pointing triangle.
            elif value < self._previous:
                suffix = " \u25BC"  # Black down-pointing triangle.
            else:
                suffix = " \u26A0"  # Warning sign.
        else:
            suffix = "  "
        return format(value, self._format) + suffix


class PlaceholderWidget(NumberTrendWidget):
    """
    A progressbar widget as a subclass of `NumberTrendWidget` that always shows the placeholder value. This is useful
    when using multiple progressbars after each other to ensure consistent formatting.
    """

    def __init__(self, format_str: str):
        super().__init__(format_str, lambda: None)
