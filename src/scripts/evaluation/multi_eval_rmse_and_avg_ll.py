import os
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from typing import Final

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from scripts.evaluation.eval_rmse_and_avg_ll import make_eval_experiment


result_dir: Final[str] = "data/temp/results"

metric_keys = ("Train RMSE", "Test RMSE", "Train Avg. LL", "Test Avg. LL")


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--start", required=True, type=int, help="First experiment ID to load (inclusive).")
    parser.add_argument("-e", "--end", required=True, type=int, help="Last experiment ID to load (inclusive).")
    args = parser.parse_args()
    exp_id_start: int = args.start
    exp_id_end: int = args.end

    results = defaultdict(lambda: defaultdict(lambda: []))
    for exp_id in tqdm(range(exp_id_start, exp_id_end + 1)):
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                ex, config = make_eval_experiment(["-d", str(exp_id)])
                run = ex.run()
        dataset_name = config["dataset"]["name"]
        for metric_key, metric in zip(metric_keys, run.result):
            results[dataset_name][metric_key].append(metric)

    table = []
    for dataset_name, metrics in sorted(results.items()):
        row = []
        for metric_key in metric_keys:
            values = np.asarray(metrics[metric_key])
            mean = values.mean()
            std = values.std() / np.sqrt(len(values))
            row.append(f"{np.format_float_positional(mean, precision=3)} ± {np.format_float_positional(std, precision=3)}")
        table.append(row)

    print(tabulate(table, headers=metric_keys, showindex=sorted(results.keys()), tablefmt="github", stralign="center"))


if __name__ == "__main__":
    main()
