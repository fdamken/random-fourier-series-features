import math
from typing import Tuple

import gpytorch
import torch
from gpytorch.models import ExactGP
from tabulate import tabulate

from ingredients import dataset
from rfsf.pre_processing.pre_processor import PreProcessor
from rfsf.util import devices
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model, load_pre_processor = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __num_samples = 5


@ex.main
def main(__figures_dir: str, __experiment_dir: str, __num_samples: int):
    pre_processor = load_pre_processor()
    model = load_model()
    pre_processor.to(devices.cpu())
    model.to(device=devices.cpu())
    model.eval()

    (train_inputs, train_targets), (test_inputs, test_targets) = dataset.load_data(device=devices.cpu())

    print(f"Computing posterior metrics for {len(train_targets)} training samples.")
    train_posterior_rmse, train_posterior_ll = _compute_rmse_and_ll(pre_processor, model, train_inputs, train_targets)
    print(f"Computing posterior metrics for {len(test_targets)} test samples.")
    test_posterior_rmse, test_posterior_ll = _compute_rmse_and_ll(pre_processor, model, test_inputs, test_targets)

    print(
        tabulate(
            [[train_posterior_rmse, train_posterior_ll], [test_posterior_rmse, test_posterior_ll]],
            headers=("RMSE", "Avg. LL"),
            showindex=("Train", "Test"),
            tablefmt="github",
        )
    )


@torch.no_grad()
def _compute_rmse_and_ll(pre_processor: PreProcessor, model: ExactGP, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        inputs = pre_processor.transform_inputs(inputs)
        targets = pre_processor.transform_targets(targets)
        predictions = model(inputs)
        mse = ((predictions.mean - targets) ** 2).sum().item()
        ll = predictions.log_prob(targets).item()
        # Take mean of log-prob to not report dataset-size-dependent metrics.
        return math.sqrt(mse / len(targets)), ll / len(targets)


if __name__ == "__main__":
    ex.run()
