import math
import os
import os.path as osp
from typing import Tuple

import gpytorch
import numpy as np
import torch
from gpytorch.models import ExactGP
from sacred import Experiment
from tabulate import tabulate

from ingredients import dataset
from rfsf.pre_processing.pre_processor import PreProcessor
from rfsf.util import devices
from scripts.util.sacred_util import load_experiment


def evaluate(pre_processor: PreProcessor, model: ExactGP, *, skip_training_evaluation: bool = False, device: torch.device = devices.cpu()) -> Tuple[float, float, float, float]:
    pre_processor.to(device)
    model.to(device=device)
    model.eval()

    (train_inputs, train_targets), (test_inputs, test_targets) = dataset.load_data(device=device)

    if skip_training_evaluation:
        train_posterior_rmse, train_posterior_ll = np.nan, np.nan
    else:
        print(f"Computing posterior metrics for {len(train_targets)} training samples.")
        train_posterior_rmse, train_posterior_ll = _compute_rmse_and_ll(pre_processor, model, train_inputs, train_targets)

    print(f"Computing posterior metrics for {len(test_targets)} test samples.")
    test_posterior_rmse, test_posterior_ll = _compute_rmse_and_ll(pre_processor, model, test_inputs, test_targets)

    return train_posterior_rmse, test_posterior_rmse, train_posterior_ll, test_posterior_ll


@torch.no_grad()
def _compute_rmse_and_ll(pre_processor: PreProcessor, model: ExactGP, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        transformed_inputs = pre_processor.transform_inputs(inputs)
        transformed_targets = pre_processor.transform_targets(targets)
        predictive_distribution = model(transformed_inputs)
        inverse_transformed_mean = pre_processor.inverse_transform_targets(predictive_distribution.mean)
        jittered_covariance_matrix = predictive_distribution.covariance_matrix + torch.eye(predictive_distribution.covariance_matrix.shape[0]) * 1e-4
        inverse_transformed_cov = pre_processor.inverse_transform_covariance_matrix(jittered_covariance_matrix)
        inverse_transformed_predictive_distribution = torch.distributions.MultivariateNormal(inverse_transformed_mean, inverse_transformed_cov)
        mse = ((inverse_transformed_mean - targets) ** 2).sum().item()
        ll = inverse_transformed_predictive_distribution.log_prob(transformed_targets).item()
        # Take mean of log-prob to not report dataset-size-dependent metrics.
        return math.sqrt(mse / len(targets)), ll / len(targets)


def make_eval_experiment(args=None) -> Tuple[Experiment, dict]:
    ex, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor = load_experiment(args)

    @ex.main
    def main(_log, __experiment_dir: str) -> Tuple[float, float, float, float]:
        train_posterior_rmse, test_posterior_rmse, train_posterior_ll, test_posterior_ll = evaluate(load_pre_processor(), load_model())

        print(
            tabulate(
                [[train_posterior_rmse, train_posterior_ll], [test_posterior_rmse, test_posterior_ll]],
                headers=("RMSE", "Avg. LL"),
                showindex=("Train", "Test"),
                tablefmt="github",
            )
        )

        eval_file = f"{__experiment_dir}/{osp.basename(__file__).replace('.py', '')}.csv"
        if osp.exists(eval_file):
            assert osp.isfile(eval_file), f"{eval_file = } exists, but is not a file"
            os.remove(eval_file)
        with open(eval_file, "w") as f:
            f.write(f"{train_posterior_rmse},{test_posterior_rmse},{train_posterior_ll},{test_posterior_ll}")

        return train_posterior_rmse, test_posterior_rmse, train_posterior_ll, test_posterior_ll

    return ex, load_config()


if __name__ == "__main__":
    make_eval_experiment()[0].run()
