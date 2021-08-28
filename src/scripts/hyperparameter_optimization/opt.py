import warnings
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import optuna
from gpytorch.utils.errors import NotPSDError
from optuna import Trial

from experiments.experiment import make_experiment
from scripts.evaluation.eval_rmse_and_avg_ll import evaluate
from scripts.util.sacred_util import _load_pickle


class CUDAOutOfMemoryError(RuntimeError):
    pass


def objective(trial: Trial, dataset: str, model_name: str, max_iter: int, learning_rate: float) -> float:
    config = {
        "dataset": {"name": dataset},
        "optimizer_kwargs": {"lr": learning_rate},
        "max_iter": max_iter,
        "model_kwargs": {
            "num_harmonics": trial.suggest_int(name="num_harmonics", low=1, high=128),
            "half_period": trial.suggest_uniform(name="half_period", low=1e-2, high=1e1),
        },
    }

    try:
        # Training.
        print(f"Training with {max_iter=}.")
        ex = make_experiment(False)
        run = ex.run(named_configs=[model_name], config_updates=config)

        # Evaluation.
        basedir = run.observers[0].dir
        pre_processor = _load_pickle(f"{basedir}/pre_processor.pkl")
        model = _load_pickle(f"{basedir}/model-final.pkl")
        return evaluate(pre_processor, model, skip_training_evaluation=True)[1]
    except RuntimeError as e:
        if hasattr(e, "args") and type(e.args) is tuple and e.args[0].startswith("CUDA out of memory"):
            warnings.warn(f"CUDA out of memory for config {config}. Pruning trial.")
            raise CUDAOutOfMemoryError()
        raise e


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str, help="Dataset to optimize hyper-parameters on.")
    parser.add_argument("-c", "--model", required=True, type=str, help="Name of the model (in the form of a Sacred named config) to optimize.")
    parser.add_argument("-n", "--max_iter", required=True, type=int, help="Maximum number of iterations to perform per trial.")
    parser.add_argument("-l", "--learning_rate", default=0.01, type=float, help="Learning rate to use.")
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model
    max_iter = args.max_iter
    learning_rate = args.learning_rate

    study = optuna.create_study(
        study_name=f"{dataset}_{model_name}_{max_iter}MaxIter_{learning_rate}LR_{datetime.now().isoformat()}",
        direction="minimize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(
        partial(objective, dataset=dataset, model_name=model_name, max_iter=max_iter, learning_rate=learning_rate),
        n_trials=100,
        catch=(NotPSDError, CUDAOutOfMemoryError),
    )
    print("Best parameters:", study.best_params)


if __name__ == "__main__":
    main()
