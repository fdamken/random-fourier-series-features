from argparse import ArgumentParser
from typing import List

from experiments.experiment import make_experiment


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment_args", required=True, type=str, help="Parameters to pass to the experiment via run_commandline.")
    parser.add_argument("-d", "--datasets", required=True, type=str, help="Comma-separated list of datasets to evaluate.")
    parser.add_argument("-n", "--num_seeds", default=5, type=int, help="Number of seeds to evaluate.")
    parser.add_argument("-m", "--num_split_seeds", default=5, type=int, help="Number of train/test-split seeds to evaluate per seed.")
    return parser


def run_experiment(seed: int, split_seed: int, dataset: str, experiment_args: List[str]):
    args = ["dummy"]
    if len(experiment_args) > 1:
        args += experiment_args
    else:
        args += ["with"]
    args += [
        f"seed={seed}",
        f"dataset.name={dataset}",
        f"dataset.train_test_split_seed={split_seed}",
    ]
    print(f"Running experiment with the arguments {args}.")
    make_experiment(log_to_wandb=False).run_commandline(args)


def main() -> None:
    args = get_parser().parse_args()
    datasets = args.datasets.split(",")
    num_seeds = args.num_seeds
    num_split_seeds = args.num_split_seeds
    experiment_args = args.experiment_args.split(" ")

    for seed in range(num_seeds):
        for split_seed in range(num_split_seeds):
            for dataset in datasets:
                run_experiment(seed, split_seed, dataset, experiment_args)


if __name__ == "__main__":
    main()
