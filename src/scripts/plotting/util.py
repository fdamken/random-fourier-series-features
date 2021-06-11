from typing import List

import numpy as np
from matplotlib import colors
from matplotlib.figure import Figure


# noinspection PyPep8Naming,SpellCheckingInspection
class AccumulativeNormalization(colors.Normalize):
    def autoscale(self, A):
        self._update(A)

    def autoscale_None(self, A):
        self._update(A)

    def _update(self, A):
        vmin = np.min(A)
        if self.vmin is None or self.vmin > vmin:
            self.vmin = vmin

        vmax = np.max(A)
        if self.vmax is None or self.vmax < vmax:
            self.vmax = vmax


def savefig(fig: Figure, path: str, filename: str, *, formats: List[str] = None) -> Figure:
    if formats is None:
        formats = ["pdf", "pgf", "png"]
    for fmt in formats:
        fig.savefig(f"{path}/{filename}.{fmt}")
    return fig
