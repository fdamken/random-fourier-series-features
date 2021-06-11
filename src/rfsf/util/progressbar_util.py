from typing import Callable, Optional

import numpy as np
import progressbar


class NumberTrendWidget(progressbar.Widget):
    def __init__(self, format_str: str, observable: Callable[[], Optional[float]]):
        self._format = format_str
        self._observable = observable
        self._placeholder = " " * (len(self._format % 0.0) + 2)
        self._previous = None

    def update(self, _):
        value = self._observable()
        if self._previous is not None and value is not None:
            if np.isclose(value, self._previous):
                suffix = " \u25B6"  # Black right-pointing triangle.
            elif value > self._previous:
                suffix = " \u25B2"  # Black up-pointing triangle.
            elif value < self._previous:
                suffix = " \u25BC"  # Black down-pointing triangle.
            else:
                suffix = " \u26A0"  # Warning sign.
        else:
            suffix = "  "
        self._previous = value
        return self._placeholder if value is None else (self._format % value + suffix)


class PlaceholderWidget(NumberTrendWidget):
    def __init__(self, format_str: str):
        super().__init__(format_str, lambda: None)
