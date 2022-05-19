import os
from typing import Final, List, Optional
import tikzplotlib
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

HIDE_DEBUG_INFO: Final[bool] = os.environ.get("HIDE_DEBUG_INFO") is not None

# noinspection PyPep8Naming,SpellCheckingInspection
class AccumulativeNormalization(colors.Normalize):
    def autoscale(self, A):
        self._update(A)

    def autoscale_None(self, A):
        self._update(A)

    def _update(self, A):
        if A.size <= 0:
            return

        vmin = np.min(A)
        if self.vmin is None or self.vmin > vmin:
            self.vmin = vmin

        vmax = np.max(A)
        if self.vmax is None or self.vmax < vmax:
            self.vmax = vmax


def savefig(fig: plt.Figure, path: str, filename: str, *, formats: List[str] = None) -> plt.Figure:
    if formats is None:
        formats = ["pdf", "pgf", "png", "tikz"]
    plt.tight_layout()
    for fmt in formats:
        file = f"{path}/{filename}.{fmt}"
        if fmt == "tikz":
            tikzplotlib.clean_figure(fig)
            tikzplotlib.save(file, figure=fig)
        else:
            fig.savefig(file)
    return fig


def show_debug_info(fig: plt.Figure, run: dict, experiment_dir: str):
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


def make_colored_line_collection(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    cmap: colors.Colormap = plt.get_cmap("copper"),
    norm: plt.Normalize = plt.Normalize(0.0, 1.0),
    linewidth: int = 3,
    alpha: float = 1.0,
) -> LineCollection:
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
