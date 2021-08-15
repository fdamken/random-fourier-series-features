import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

from scripts.plotting.util import savefig


project = "tuda-ias-rfsf/random-fourier-series-features"
run_ids = {
    "both": "6p3sgyx2",
    "only_amplitudes": "1adn2pgp",
    "only_phases": "24g80t1l",
}

lut_category = {
    "amplitudes_sqrt": "Amplitudes (Sqrt Space)",
    "phases": "Phases",
}
lut_runs = {
    "only_phases": "Optimizing only Phases",
    "only_amplitudes": "Optimizing only Amplitudes",
    "both": "Optimizing Everything",
}
lut_run_colors = {
    "only_phases": "tab:blue",
    "only_amplitudes": "tab:orange",
    "both": "tab:green",
}
lut_run_cmaps = {
    "only_phases": "Blues",
    "only_amplitudes": "Oranges",
    "both": "Greens",
}


def plot_line_plot(metrics: Dict[str, Dict[str, Dict[int, List[Tuple[int, float]]]]], runs: Dict[str, any]) -> None:
    fig, axss = plt.subplots(ncols=len(runs), nrows=len(metrics), figsize=(5 * len(runs), len(metrics) * 5), sharex="all", sharey="row")
    for y_idx, (axs, (category, cat_metrics)) in enumerate(zip(axss, metrics.items())):
        first = True
        for ax, (run, key_metrics) in zip(axs, cat_metrics.items()):
            for index, values in sorted(key_metrics.items()):
                ax.plot(*np.asarray(values).T, label=index)
            if y_idx == len(axss) - 1:
                ax.set_xlabel("Iteration")
            if first:
                ax.set_ylabel(lut_category[category])
            if y_idx == 0:
                ax.set_title(lut_runs[run])
            ax.legend(ncol=3)
            first = False
    plt.tight_layout()
    fig.show()


def plot_scatter_matrix(metrics: Dict[str, Dict[str, Dict[int, List[Tuple[int, float]]]]]) -> plt.Figure:
    metrics_list_dict = defaultdict(lambda: {})
    for category, cat_metrics in metrics.items():
        for run, run_metrics in cat_metrics.items():
            for index, values in run_metrics.items():
                metrics_list_dict[f"{category} [{index}]"][run] = np.asarray(sorted(values))[:, 1]
    metrics_list = sorted(metrics_list_dict.items())

    grid_size = len(metrics_list)
    fig, axss = plt.subplots(ncols=grid_size, nrows=grid_size, figsize=(grid_size * 5, grid_size * 5))
    with tqdm(total=len(metrics_list) ** 2, desc="Plotting") as pbar:
        for x, (axs, (x_param_name, x_metrics)) in enumerate(zip(axss.T, metrics_list)):
            for y, (ax, (y_param_name, y_metrics)) in enumerate(zip(axs, metrics_list)):
                assert x_metrics.keys() == y_metrics.keys()
                for run in x_metrics.keys():
                    ax.plot(x_metrics[run], y_metrics[run], color=lut_run_colors[run], label=lut_runs[run], linewidth=1, zorder=1)
                    ax.scatter(x_metrics[run][0], y_metrics[run][0], color=lut_run_colors[run], edgecolor="black", zorder=2)
                    ax.scatter(x_metrics[run][-1], y_metrics[run][-1], color=lut_run_colors[run], edgecolor="black", marker="*", s=8 ** 2, zorder=2)
                ax.set_xlabel(x_param_name)
                ax.set_ylabel(y_param_name)
                ax.legend()
                pbar.update()
    plt.tight_layout()
    return fig


def main():
    api = wandb.Api()

    print("Loading data.")
    runs = {run: api.run(f"{project}/{id}") for run, id in run_ids.items()}
    metrics = defaultdict(lambda: {run: defaultdict(lambda: []) for run in run_ids.keys()})
    for run, run_data in runs.items():
        for row in run_data.history(pandas=False):
            for metric_key, metric_value in row.items():
                match = re.match(r"^parameters/cov_module\.([\w_]+)\[(\d+)]$", metric_key)
                if not match:
                    continue
                category, index = match.group(1), int(match.group(2))
                metrics[category][run][index].append((row["_step"], metric_value))

    fig = plot_scatter_matrix(metrics)
    print("Saving figure.")
    savefig(fig, "data/temp/figures", "amplitudes_phases_path", formats=["png"])


if __name__ == "__main__":
    main()
