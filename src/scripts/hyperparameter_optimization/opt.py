import warnings
from datetime import datetime
from typing import Final

import optuna
from gpytorch.utils.errors import NotPSDError
from optuna import Trial

from experiments.experiment import make_experiment
from scripts.evaluation.eval_rmse_and_avg_ll import evaluate
from scripts.util.sacred_util import _load_pickle


dataset: Final[str] = "uci-concrete"
model_name: Final[str] = "rfsf_relu"
max_iter = 256


class CUDAOutOfMemoryError(RuntimeError):
    pass


def objective(trial: Trial) -> float:
    config = {
        "dataset": {"name": dataset},
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
    study = optuna.create_study(
        study_name=f"{dataset}_{model_name}_{max_iter}MaxIter_{datetime.now().isoformat()}",
        direction="minimize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=100,
        catch=(NotPSDError, CUDAOutOfMemoryError),
    )
    print(study.best_params)


if __name__ == "__main__":
    main()
