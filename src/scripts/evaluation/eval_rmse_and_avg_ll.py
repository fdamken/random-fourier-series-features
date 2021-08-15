from typing import Tuple

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
    pre_processor.to(devices.cuda())
    model.to(device=devices.cuda())

    (train_inputs, train_targets), (test_inputs, test_targets) = dataset.load_data(device=devices.cuda())

    model.eval()
    train_posterior_rmse, train_posterior_ll = _compute_rmse_and_ll(pre_processor, model, train_inputs, train_targets)
    test_posterior_rmse, test_posterior_ll = _compute_rmse_and_ll(pre_processor, model, test_inputs, test_targets)

    model.train()
    train_prior_rmse, train_prior_ll = _compute_rmse_and_ll(pre_processor, model, train_inputs, train_targets)
    test_prior_rmse, test_prior_ll = _compute_rmse_and_ll(pre_processor, model, test_inputs, test_targets)

    print(tabulate(
        [[train_posterior_rmse, train_posterior_ll, train_prior_rmse, train_prior_ll],
         [test_posterior_rmse, test_posterior_ll, test_prior_rmse, test_prior_ll]],
        headers=("Posterior RMSE", "Posterior Avg. LL", "Prior RMSE", "Prior Avg. LL"),
        showindex=("Train", "Test"),
        tablefmt="github",
    ))


@torch.no_grad()
def _compute_rmse_and_ll(pre_processor: PreProcessor, model: ExactGP, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    train_inputs, train_targets = model.train_inputs, model.train_targets
    inputs = pre_processor.transform_inputs(inputs)
    targets = pre_processor.transform_targets(targets)
    if model.training:
        # Hack the test data as training data into the model to compute the prior.
        model.set_train_data(inputs, targets, strict=False)
    predictions = model(inputs)
    if model.training:
        # Reset the training data to clear the model up for the next evaluation.
        model.set_train_data(train_inputs, train_targets, strict=False)
    rmse = torch.sqrt(((predictions.mean - targets) ** 2).mean()).item()
    ll = predictions.log_prob(targets).item() / len(targets)  # Take mean of log-prob to not report size-dependent metrics.
    return rmse, ll


if __name__ == "__main__":
    ex.run()
