from typing import List

from matplotlib.figure import Figure


def savefig(fig: Figure, path: str, filename: str, *, formats: List[str] = None) -> Figure:
    if formats is None:
        formats = ["pdf", "pgf", "png"]
    for fmt in formats:
        fig.savefig(f"{path}/{filename}.{fmt}")
    return fig
