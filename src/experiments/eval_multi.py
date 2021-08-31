import warnings
from argparse import ArgumentParser
from typing import List

from experiments.experiment import make_experiment


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment_args", required=True, type=str, help="Parameters to pass to the experiment via run_commandline.")
    parser.add_argument("-d", "--datasets", required=True, type=str, help="Comma-separated list of datasets to evaluate.")
    parser.add_argument("-n", "--num_eval", default=5, type=int, help="Number of seeds to evaluate.")
    return parser


def run_experiment(seed: int, uci_split_index: int, dataset: str, experiment_args: List[str]):
    args = ["dummy"]
    if len(experiment_args) > 1:
        args += experiment_args
    else:
        args += ["with"]
    args += [
        f"seed={seed}",
        f"dataset.name={dataset}",
        f"dataset.uci_split_index={uci_split_index}",
    ]
    print(f"Running experiment with the arguments {args}.")
    make_experiment(log_to_wandb=False).run_commandline(args)


def main() -> None:
    args = get_parser().parse_args()
    datasets = args.datasets.split(",")
    num_eval = args.num_eval
    experiment_args = args.experiment_args.split(" ")

    for seed in range(num_eval):
        for dataset in datasets:
            for uci_split_index in range(20):
                try:
                    run_experiment(seed, uci_split_index, dataset, experiment_args)
                except:
                    # This is sad. But better than crashing the whole evaluation.
                    warnings.warn(f"Experiment for seed {seed}, dataset {dataset}, and split index {uci_split_index} failed. Ignoring.")


if __name__ == "__main__":
    main()
