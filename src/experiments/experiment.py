import os
from logging import Logger
from typing import Any, ClassVar, List, Optional

import numpy as np
import sklearn.utils
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ExactGP
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from experiments.models.rbf_gp import RBFGP
from experiments.models.rff_gp import RFFGP
from experiments.models.rfsf_random_gp import RFSFRandomGP
from experiments.models.rfsf_relu_gp import RFSFReLUGP
from experiments.models.rfsf_single_harmonic_gp import RFSFSingleHarmonicGP
from experiments.util.sacred_util import add_pickle_artifact
from experiments.util.wandb_observer import WandbObserver
from ingredients import dataset
from ingredients.dataset import dataset_ingredient
from rfsf.kernel.rfsf_kernel import RFSFKernel
from rfsf.pre_processing.pca_whitening import PCAInputWhitening
from rfsf.pre_processing.pre_processor import PreProcessor
from rfsf.util import devices
from rfsf.util.axial_iteration_lr import AxialIterationLR
from rfsf.util.constant_lr import ConstantLR
from rfsf.util.mock_lr import MockLR
from rfsf.util.tensor_util import apply_parameter_name_selector, gen_index_iterator, split_parameter_groups


def make_experiment(log_to_wandb: bool, slurm_array_job_id: Optional[str], slurm_array_job_index: Optional[str]) -> Experiment:
    is_slurm = slurm_array_job_id is not None and slurm_array_job_index is not None
    ex = Experiment(ingredients=[dataset_ingredient])
    tags = []
    storage_dir = "data/temp/results"
    if is_slurm:
        storage_dir += f"/slurm/{slurm_array_job_id}_{slurm_array_job_index}"
        tags.append("slurm")
    ex.observers.append(FileStorageObserver(storage_dir))
    if log_to_wandb:
        ex.observers.append(WandbObserver(project="random-fourier-series-features", entity="tuda-ias-rfsf", tags=tags))

    # noinspection PyUnusedLocal
    @ex.config
    def default_config():
        seed = 42
        likelihood_class = GaussianLikelihood
        likelihood_kwargs = {}
        pre_processor_class = PCAInputWhitening
        pre_processor_kwargs = {}
        model_class = ExactGP  # Has to be overwritten by named configs.
        model_kwargs = {}
        optimizer_class = Adam
        optimizer_kwargs = {"lr": 0.01}
        optimizer_alternate_parameters = [["all"]]
        lr_scheduler_class = ExponentialLR
        lr_scheduler_kwargs = {"gamma": 0.999}
        max_iter = 1000
        batch_size = 1000
        save_model_every_n_iterations = 10
        log_parameter_values = False
        log_parameter_grad_values = False

    # noinspection PyUnusedLocal
    @ex.named_config
    def rfsf_random():
        model_class = RFSFRandomGP

        model_kwargs = dict(
            num_samples=2500,
            num_harmonics=16,
            half_period=50.0,
            optimize_amplitudes=True,  # Ignored iff use_sine_cosine_form is `True`.
            optimize_phases=True,  # Ignored iff use_sine_cosine_form is `True`.
            optimize_half_period=True,
            use_sine_cosine_form=True,
            use_ard=True,
        )

    # noinspection PyUnusedLocal
    @ex.named_config
    def rfsf_relu():
        model_class = RFSFReLUGP

        model_kwargs = dict(
            num_samples=2500,
            num_harmonics=16,
            half_period=50.0,
            optimize_amplitudes=True,  # Ignored iff use_sine_cosine_form is `True`.
            optimize_phases=True,  # Ignored iff use_sine_cosine_form is `True`.
            optimize_half_period=True,
            use_sine_cosine_form=True,
            use_ard=True,
        )

    # noinspection PyUnusedLocal
    @ex.named_config
    def rfsf_single_harmonic():
        model_class = RFSFSingleHarmonicGP

        model_kwargs = dict(
            num_samples=2500,
            num_harmonics=16,
            half_period=50.0,
            optimize_amplitudes=True,  # Ignored iff use_sine_cosine_form is `True`.
            optimize_phases=True,  # Ignored iff use_sine_cosine_form is `True`.
            optimize_half_period=True,
            use_sine_cosine_form=True,
            use_ard=True,
        )

    # noinspection PyUnusedLocal
    @ex.named_config
    def rbf():
        model_class = RBFGP
        model_kwargs = dict(use_ard=True)

    # noinspection PyUnusedLocal
    @ex.named_config
    def rff():
        model_class = RFFGP

        model_kwargs = dict(
            num_samples=2500,
            use_ard=True,
        )

    @ex.main
    def main(
        likelihood_class: ClassVar[Likelihood],
        likelihood_kwargs: dict,
        pre_processor_class: ClassVar[PreProcessor],
        pre_processor_kwargs: dict,
        model_class: ClassVar[ExactGP],
        model_kwargs: dict,
        optimizer_class: ClassVar[Optimizer],
        optimizer_kwargs: dict,
        optimizer_alternate_parameters: List[List[str]],
        lr_scheduler_class: ClassVar[Any],
        lr_scheduler_kwargs: dict,
        max_iter: int,
        batch_size: int,
        save_model_every_n_iterations: int,
        log_parameter_values: bool,
        log_parameter_grad_values: bool,
        _run: Run,
        _log: Logger,
    ):
        (train_inputs, train_targets), _ = dataset.load_data(device=devices.cuda(), double_precision=True)

        pre_processor: PreProcessor = pre_processor_class(**pre_processor_kwargs)
        pre_processor.fit(train_inputs, train_targets)
        train_inputs = pre_processor.transform_inputs(train_inputs)
        train_targets = pre_processor.transform_targets(train_targets)

        # Add the pre-processor as an artifact directly after fitting it such that the buffers are initialized.
        add_pickle_artifact(_run, pre_processor, "pre_processor", device=devices.cuda())

        likelihood: Likelihood = likelihood_class(**likelihood_kwargs).double()
        model: ExactGP = model_class(train_inputs, train_targets, likelihood, **model_kwargs).double()

        likelihood.to(devices.cuda())
        model.to(devices.cuda())

        likelihood.train()
        model.train()

        _log.info(
            f"Training a GP with {sum(param.numel() for param in model.parameters() if param.requires_grad)} parameters.\n"
            + f"  - Learnable Parameters: {', '.join(f'{name} [{param.numel()} params, shape {tuple(param.shape)}]' for name, param in model.named_parameters() if param.requires_grad)}\n"
            + f"  - Non-Learn Parameters: {', '.join(f'{name} shape {tuple(param.shape)}' for name, param in model.named_parameters() if not param.requires_grad)}",
        )

        avg_loss = np.nan
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
        with tqdm(total=max_iter, desc="Optimization") as pbar:
            for step in range(max_iter):
                optimizer = optimizers[step % len(optimizers)]

                learning_rates = scheduler.get_last_lr()

                avg_loss = 0.0
                batch_count = 0
                # Computing the predictive mean/variance has to be done on the complete training data.
                train_predictions = model(train_inputs)
                # Use batches for computing the likelihood and computing the updates as the gradient might not fit on the GPU.
                for batch_slice in sklearn.utils.gen_batches(len(train_inputs), batch_size):
                    optimizer.zero_grad()
                    loss = -mll(train_predictions[batch_slice], train_targets[batch_slice].squeeze())  # FIXME: Squeezing might cause issues for multi-output.
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
                    batch_count += 1
                avg_loss /= batch_count
                scheduler.step()

                noise = model.likelihood.noise.item()
                parameters = [p for p in model.parameters() if p.grad is not None]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(devices.cuda()) for p in parameters]))
                cov_module = getattr(model, "cov_module", None)
                if isinstance(cov_module, ScaleKernel):
                    cov_module = cov_module.base_kernel
                if isinstance(cov_module, RFSFKernel):
                    half_period = cov_module.half_period.item()
                else:
                    half_period = None

                _run.log_scalar("loss", avg_loss, step=step)
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
                if half_period is not None:
                    _run.log_scalar("half_period", half_period, step=step)
                if step % save_model_every_n_iterations == 0:
                    add_pickle_artifact(_run, model, f"model-{step}", device=devices.cuda())

                pbar.set_postfix({"loss": avg_loss})
                pbar.update()

        add_pickle_artifact(_run, model, f"model-final", device=devices.cuda())

        return avg_loss

    return ex


if __name__ == "__main__":
    make_experiment(os.environ.get("NO_WANDB") is None, os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")).run_commandline()
