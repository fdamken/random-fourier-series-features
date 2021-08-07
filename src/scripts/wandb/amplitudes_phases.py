import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import wandb


project = "tuda-ias-rfsf/random-fourier-series-features"
lut_category = {
    "amplitudes_sqrt": "Amplitudes (Sqrt Space)",
    "phases": "Phases",
}
lut_key = {
    "only_phases": "Optimizing only Phases",
    "only_amplitudes": "Optimizing only Amplitudes",
    "both": "Optimizing Everything",
}


def main():
    api = wandb.Api()

    run_ids = {
        "both": "6p3sgyx2",
        "only_amplitudes": "1adn2pgp",
        "only_phases": "24g80t1l",
    }
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
                ax.set_title(lut_key[run])
            ax.legend(ncol=3)
            first = False
    plt.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()
