from typing import Optional, Tuple

import gpytorch
import torch
from matplotlib import cycler
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn

from rfsf.util.tensor_util import to_numpy


sample_color_cycler = cycler(color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])


def plot_process(
    model: nn.Module,
    data: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    num_samples: int,
    title: str,
    *,
    legend: bool = True,
    legend_loc: Optional[str] = "lower left",
    y_lim: Optional[Tuple[float, float]] = None,
) -> Figure:
    (train_x, train_y), (test_x, test_y) = data

    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(test_x)
    fig, ax = plt.subplots()
    lower, upper = observed_pred.confidence_region()
    ax.scatter(to_numpy(train_x), to_numpy(train_y), color="black", marker="*", s=100, label="Observed Data", zorder=3)
    ax.plot(to_numpy(test_x), to_numpy(test_y), color="black", label="True Func.", zorder=0)
    for _, c in zip(range(num_samples), sample_color_cycler):
        ax.plot(to_numpy(test_x), to_numpy(observed_pred.sample()), color=c["color"], alpha=0.5, zorder=2)
    ax.fill_between(to_numpy(test_x), to_numpy(lower), to_numpy(upper), color="tab:blue", alpha=0.2, label=r"Mean $\pm$ Confidence", zorder=1)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_title(title)
    if legend:
        ax.legend(loc=legend_loc)
    return fig
