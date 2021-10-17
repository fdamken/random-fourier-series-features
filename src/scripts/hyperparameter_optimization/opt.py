import warnings
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import optuna
from gpytorch.utils.errors import NotPSDError

from experiments.experiment import make_experiment
from scripts.evaluation.eval_rmse_and_avg_ll import evaluate
from scripts.util.sacred_util import _load_pickle


class CUDAOutOfMemoryError(RuntimeError):
    pass


def evaluate_dataset(config: dict, dataset: str, model_name: str, pre_processing: str, optimize_rmse_not_likelihood: bool) -> float:
    config["dataset"] = {"name": dataset}
    try:
        # Training.
        ex = make_experiment(False)
        run = ex.run(named_configs=[model_name, pre_processing], config_updates=config)

        # Evaluation.
        basedir = run.observers[0].dir
        pre_processor = _load_pickle(f"{basedir}/pre_processor.pkl")
        model = _load_pickle(f"{basedir}/model-final.pkl")
        _, test_posterior_rmse, _, test_posterior_ll = evaluate(pre_processor, model, skip_training_evaluation=True)
        return test_posterior_rmse if optimize_rmse_not_likelihood else test_posterior_ll
    except RuntimeError as e:
        if hasattr(e, "args") and type(e.args) is tuple and e.args[0].startswith("CUDA out of memory"):
            warnings.warn(f"CUDA out of memory for config {config}. Pruning trial.")
            raise CUDAOutOfMemoryError()
        raise e


def objective(trial: optuna.Trial, datasets: str, model_name: str, max_iter: int, learning_rate: float, optimize_rmse_not_likelihood: bool) -> float:
    num_harmonics = trial.suggest_int(name="num_harmonics", low=1, high=32)
    half_period = trial.suggest_float(name="half_period", low=0.1, high=10)
    use_ard = trial.suggest_categorical(name="use_ard", choices=[True, False])
    pre_processing = trial.suggest_categorical(name="pre_processing", choices=["no_pre_processing", "standardization", "pca_whitening"])

    config = {
        "optimizer_kwargs": {"lr": learning_rate},
        "max_iter": max_iter,
        "model_kwargs": {"num_harmonics": num_harmonics, "half_period": half_period, "use_ard": use_ard},
    }

    value = 0.0
    for dataset in datasets:
        value += evaluate_dataset(config, dataset, model_name, pre_processing, optimize_rmse_not_likelihood)
    return value / len(datasets)


def main() -> None:
    optimize_rmse_not_likelihood = False  # True <=> Optimize RMSE;  False <=> Optimize LL

    parser = ArgumentParser()
    parser.add_argument("-d", "--datasets", required=True, type=str, help="Datasets to optimize the hyper-parameters on.")
    parser.add_argument("-c", "--model", required=True, type=str, help="Name of the model (in the form of a Sacred named config) to optimize.")
    parser.add_argument("-n", "--max_iter", required=True, type=int, help="Maximum number of iterations to perform per trial.")
    parser.add_argument("-k", "--num_trials", default=100, type=int, help="Number of Optuna trials.")
    parser.add_argument("-l", "--learning_rate", default=0.01, type=float, help="Learning rate to use.")
    args = parser.parse_args()
    datasets = args.datasets.split(",")
    model_name = args.model
    max_iter = args.max_iter
    learning_rate = args.learning_rate
    num_trials = args.num_trials

    study = optuna.create_study(
        study_name=f"{','.join(datasets)}_{model_name}_{max_iter}MaxIter_{learning_rate}LR_{datetime.now().isoformat()}",
        direction="minimize" if optimize_rmse_not_likelihood else "maximize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(
        partial(objective, datasets=datasets, model_name=model_name, max_iter=max_iter, learning_rate=learning_rate, optimize_rmse_not_likelihood=optimize_rmse_not_likelihood),
        n_trials=num_trials,
        catch=(NotPSDError, CUDAOutOfMemoryError),
    )
    print("Best parameters:", study.best_params)


if __name__ == "__main__":
    main()
