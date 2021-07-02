from logging import Logger
from typing import Any, ClassVar, List

import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ExactGP
from progressbar import Bar, ETA, Percentage, ProgressBar
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run
from torch.optim import Adam, LBFGS, Optimizer

from experiments.models.rfsf_random_gp import RFSFRandomGP
from experiments.models.rfsf_relu_gp import RFSFReLUGP
from experiments.models.scaled_rbf_gp import ScaledRBFGP
from experiments.util.sacred_util import add_pickle_artifact
from experiments.util.wandb_observer import WandbObserver
from ingredients import dataset
from ingredients.dataset import dataset_ingredient
from rfsf.util import devices
from rfsf.util.axial_iteration_lr import AxialIterationLR
from rfsf.util.constant_lr import ConstantLR
from rfsf.util.mock_lr import MockLR
from rfsf.util.progressbar_util import NumberTrendWidget
from rfsf.util.tensor_util import apply_parameter_name_selector, gen_index_iterator, pickle_str, split_parameter_groups


ex = Experiment(ingredients=[dataset_ingredient])
ex.observers.append(FileStorageObserver("data/temp/results"))
ex.observers.append(WandbObserver(project="random-fourier-series-features", entity="tuda-ias-rfsf"))


# noinspection PyUnusedLocal
@ex.config
def default_config():
    seed = 42
    likelihood_class = GaussianLikelihood
    likelihood_kwargs = {}
    model_class = ExactGP  # Has to be overwritten by named configs.
    model_kwargs = {}
    optimizer_class = Adam
    optimizer_kwargs = {"lr": 0.01}
    optimizer_alternate_parameters = [["all"]]
    lr_scheduler_class = ConstantLR
    lr_scheduler_kwargs = {}
    max_iter = 20000
    log_model_state_every_n_iterations = 100
    log_parameter_values = True
    log_parameter_grad_values = True


# noinspection PyUnusedLocal
@ex.named_config
def axial_iteration_opt():
    optimizer_alternate_parameters = [["all", "!cov_module.phases"], ["all", "!cov_module.amplitudes_sqrt"]]


# noinspection PyUnusedLocal
@ex.named_config
def axial_iteration_lr():
    lr_scheduler_class = AxialIterationLR
    lr_scheduler_kwargs = {
        "axial_iteration_over": ["cov_module.amplitudes_sqrt", "cov_module.phases"],
        "epoch_inverse_scale": 1000,
    }


# noinspection PyUnusedLocal
@ex.named_config
def lbfgs():
    optimizer_class = LBFGS
    max_iter = 1000


# noinspection PyUnusedLocal
@ex.named_config
def scaled_rbf():
    model_class = ScaledRBFGP


# noinspection PyUnusedLocal
@ex.named_config
def rfsf_random():
    model_class = RFSFRandomGP
    model_kwargs = dict(
        num_samples=5000,
        num_harmonics=16,
        half_period=1.0,
        optimize_amplitudes=True,
        optimize_phases=True,
    )


# noinspection PyUnusedLocal
@ex.named_config
def rfsf_relu():
    model_class = RFSFReLUGP
    model_kwargs = dict(
        num_samples=5000,
        num_harmonics=8,
        half_period=1.0,
        optimize_amplitudes=True,
        optimize_phases=True,
    )


@ex.automain
def main(
    likelihood_class: ClassVar[Likelihood],
    likelihood_kwargs: dict,
    model_class: ClassVar[ExactGP],
    model_kwargs: dict,
    optimizer_class: ClassVar[Optimizer],
    optimizer_kwargs: dict,
    optimizer_alternate_parameters: List[List[str]],
    lr_scheduler_class: ClassVar[Any],
    lr_scheduler_kwargs: dict,
    max_iter: int,
    log_model_state_every_n_iterations: int,
    log_parameter_values: bool,
    log_parameter_grad_values: bool,
    _run: Run,
    _log: Logger,
):
    (train_inputs, train_targets), _ = dataset.load_data(device=devices.cuda())

    likelihood: Likelihood = likelihood_class(**likelihood_kwargs)
    model: ExactGP = model_class(train_inputs, train_targets, likelihood, **model_kwargs)

    likelihood.to(devices.cuda())
    model.to(devices.cuda())

    likelihood.train()
    model.train()

    _log.info(
        f"Training a GP with {sum(param.numel() for param in model.parameters() if param.requires_grad)} parameters.\n"
        + f"  - Learnable Parameters: {', '.join(f'{name} [{param.numel()} params, shape {tuple(param.shape)}]' for name, param in model.named_parameters() if param.requires_grad)}\n"
        + f"  - Non-Learn Parameters: {', '.join(f'{name} shape {tuple(param.shape)}' for name, param in model.named_parameters() if not param.requires_grad)}",
    )

    loss_val = None
    noise = None

    bar = ProgressBar(
        widgets=[
            "Optimization: ",
            Percentage(),
            " ",
            Bar(),
            " ",
            ETA(),
            ";  Loss: ",
            NumberTrendWidget("6.3f", lambda: loss_val),
            ";  Noise: ",
            NumberTrendWidget("6.3f", lambda: noise),
        ],
        maxval=max_iter,
        term_width=200,
    ).start()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    parameter_group_names, opt_parameters = split_parameter_groups(model)
    if len(optimizer_alternate_parameters) > 1:
        assert lr_scheduler_class is ConstantLR, "learning rate schedulers are not supported when using multiple optimizers"
        covered_parameters = set()
        optimizers, optimizers_parameters = [], []
        for opt_parameter_selector in optimizer_alternate_parameters:
            selected_parameter_names = apply_parameter_name_selector(parameter_group_names, opt_parameter_selector)
            selected_opt_parameters = [opt_parameters[parameter_group_names.index(name)] for name in selected_parameter_names]
            optimizers.append(optimizer_class(selected_opt_parameters, **optimizer_kwargs))
            optimizers_parameters.append(selected_parameter_names)
            covered_parameters = covered_parameters.union(set(selected_parameter_names))
        assert set(parameter_group_names) == covered_parameters, "optimizer specification does not cover for all existing parameters"
        scheduler = MockLR()
    else:
        if lr_scheduler_class is AxialIterationLR:
            assert optimizer_class is not LBFGS, "L-BFGS optimization is not supported when using axial iteration learning rates; try multiple optimizers instead"
            optimizer = optimizer_class([{"params": params} for params in opt_parameters], **optimizer_kwargs)
            new_lr_scheduler_kwargs = {"parameter_group_names": parameter_group_names}
            new_lr_scheduler_kwargs.update(lr_scheduler_kwargs)
            lr_scheduler_kwargs = new_lr_scheduler_kwargs
            _log.info(f"Using axial iteration optimization with the keyword arguments {lr_scheduler_kwargs}.")
        else:
            optimizer = optimizer_class(opt_parameters, **optimizer_kwargs)
        scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        optimizers = [optimizer]
        optimizers_parameters = [parameter_group_names]
    _log.info(
        f"Using {len(optimizers_parameters)} optimizer{'s' if len(optimizers_parameters) > 1 else ''} with the following parameters:\n"
        + "\n".join(f"  - {optimizer_parameters}" for optimizer_parameters in optimizers_parameters)
    )
    for step in range(max_iter):
        optimizer = optimizers[step % len(optimizers)]

        learning_rates = scheduler.get_last_lr()

        def evaluate():
            optimizer.zero_grad()
            out = model(train_inputs)
            local_loss = -mll(out, train_targets)
            local_loss.backward()
            return local_loss

        if isinstance(optimizer, LBFGS):
            loss = optimizer.step(evaluate)
        else:
            loss = evaluate()
            optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        noise = model.likelihood.noise.item()
        parameters = [p for p in model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(devices.cuda()) for p in parameters]))

        _run.log_scalar("loss", loss_val, step=step)
        _run.log_scalar("noise", noise, step=step)
        if len(learning_rates) == 1:
            _run.log_scalar("learning_rate", learning_rates[0], step=step)
        else:
            for param_name, lr in zip(parameter_group_names, learning_rates):
                _run.log_scalar(f"learning_rate/{param_name}", lr, step=step)
        if log_parameter_values or log_parameter_grad_values:
            for param_name, param in model.named_parameters():
                if param.numel() == 1:
                    _run.log_scalar(f"parameters/{param_name}", param.item(), step=step)
                    if log_parameter_grad_values and param.requires_grad:
                        _run.log_scalar(f"parameters_grad/{param_name}", param.grad.item(), step=step)
                elif param.numel() > 1:
                    for index in gen_index_iterator(param):
                        metric_suffix = f"{param_name}[{', '.join([str(i) for i in index])}]"
                        _run.log_scalar(f"parameters/{metric_suffix}", param[index].item(), step=step)
                        if log_parameter_grad_values and param.requires_grad:
                            _run.log_scalar(f"parameters_grad/{metric_suffix}", param.grad[index].item(), step=step)
        _run.log_scalar("grad_norm", grad_norm.item(), step=step)
        if step % log_model_state_every_n_iterations == 0:
            _run.log_scalar("model_state", pickle_str(model.state_dict()), step=step)
        if step == 0:
            # Add the model as an artifact after it was first invoked as otherwise the random weights/biases are not
            # initialized and loading would fail.
            add_pickle_artifact(_run, model, "model", device=devices.cuda())

        bar.update(step)
    bar.finish()

    return {"model_state": pickle_str(model.state_dict())}
