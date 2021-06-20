import os
from typing import Final, List

import numpy as np
from matplotlib import colors
from matplotlib.figure import Figure


HIDE_DEBUG_INFO: Final[bool] = os.environ.get("HIDE_DEBUG_INFO") is not None


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


def show_debug_info(fig: Figure, run: dict, experiment_dir: str):
    repos = []
    for repo in run["experiment"]["repositories"]:
        if repo not in repos:
            repos.append(repo)
    texts = []
    for repo in repos:
        commit, dirty, url = repo["commit"], repo["dirty"], repo["url"]
        if len(repos) > 1:
            text = f"{url}@{commit}"
        else:
            text = commit
        if dirty:
            text += ", Dirty!"
        texts.append(text)
    t = fig.text(0, 0, "\n".join(texts), horizontalalignment="left", verticalalignment="bottom")
    if HIDE_DEBUG_INFO:
        t.set_color("white")

    t = fig.text(1, 0, experiment_dir, horizontalalignment="right", verticalalignment="bottom")
    if HIDE_DEBUG_INFO:
        t.set_color("white")

    return fig
