from typing import ClassVar

import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ExactGP
from progressbar import ETA, Bar, Percentage, ProgressBar
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from experiments.models.rfsf_random_gp import RFSFRandomGP
from experiments.models.rfsf_relu_gp import RFSFReLUGP
from experiments.models.scaled_rbf_gp import ScaledRBFGP
from experiments.util.sacred_util import add_pickle_artifact
from experiments.util.wandb_observer import WandbObserver
from ingredients import dataset
from ingredients.dataset import dataset_ingredient
from rfsf.util import devices
from rfsf.util.progressbar_util import NumberTrendWidget
from rfsf.util.tensor_util import pickle_str


ex = Experiment(ingredients=[dataset_ingredient])
ex.observers.append(FileStorageObserver("data/temp/results"))
ex.observers.append(WandbObserver(project="random-fourier-series-features"))


# noinspection PyUnusedLocal
@ex.config
def default_config():
    seed = 42
    learning_rate = 0.1
    max_iter = 20000
    log_model_state_every_n_iterations = 100
    likelihood_class = GaussianLikelihood
    likelihood_kwargs = {}
    model_class = ExactGP  # Has to be overwritten by named configs.
    model_kwargs = {}
    # This only affects learning when using an RFSFKernel. The maximum batch size depends on the VRAM of the GPU. On a
    # GPU with 10GiB VRAM, the maximum batch size is rougly 500.
    batch_size = 8


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
        num_harmonics=16,
        half_period=1.0,
        optimize_amplitudes=False,
        optimize_phases=True,
    )


@ex.automain
def main(
    likelihood_class: ClassVar[Likelihood],
    likelihood_kwargs: dict,
    model_class: ClassVar[ExactGP],
    model_kwargs: dict,
    max_iter: int,
    learning_rate: float,
    log_model_state_every_n_iterations: int,
    batch_size: int,
    _run: Run,
):
    (train_inputs, train_targets), _ = dataset.load_data(device=devices.cuda())

    likelihood: Likelihood = likelihood_class(**likelihood_kwargs)
    model: ExactGP = model_class(train_inputs, train_targets, likelihood, **model_kwargs)

    likelihood.to(devices.cuda())
    model.to(devices.cuda())

    likelihood.train()
    model.train()

    loss_val = None
    noise = None

    print(
        f"Training a GP with {sum(param.numel() for param in model.parameters() if param.requires_grad)} parameters.\n"
        f"  - Learnable Parameters: {', '.join(f'{name} shape {tuple(param.shape)}' for name, param in model.named_parameters() if param.requires_grad)}\n"
        f"  - Non-Learn Parameters: {', '.join(f'{name} shape {tuple(param.shape)}' for name, param in model.named_parameters() if not param.requires_grad)}"
    )

    loss_observable = lambda: loss_val
    noise_observable = lambda: noise
    bar_widgets = [
        "Optimization: ",
        Percentage(),
        " ",
        Bar(),
        " ",
        ETA(),
        ";  Loss: ",
        NumberTrendWidget("%6.3f", loss_observable),
        ";  Noise: ",
        NumberTrendWidget("%6.3f", noise_observable),
    ]
    bar = ProgressBar(widgets=bar_widgets, maxval=max_iter, term_width=200).start()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, 0.999)
    for step in range(max_iter):
        optimizer.zero_grad()
        # noinspection PyTypeChecker
        loss: torch.Tensor = 0.0
        # if hasattr(model, "cov_module") and isinstance(model.cov_module, RFSFKernel):
        #     for _ in range(batch_size):
        #         model.cov_module.clear_weights_and_phases()
        #         out = model(train_inputs)
        #         loss += -mll(out, train_targets)
        #     loss /= batch_size
        # else:
        out = model(train_inputs)
        loss += -mll(out, train_targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        noise = model.likelihood.noise.item()
        learning_rate = scheduler.get_last_lr()[0]
        parameters = [p for p in model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(devices.cuda()) for p in parameters]))

        _run.log_scalar("loss", loss_val, step=step)
        _run.log_scalar("noise", noise, step=step)
        _run.log_scalar("learning_rate", learning_rate, step=step)
        _run.log_scalar("grad_norm", grad_norm.item(), step=step)
        if step % log_model_state_every_n_iterations == 0:
            _run.log_scalar("model_state", pickle_str(model.state_dict()), step=step)
        if step == 0:
            # Add the model as an artifact after it was first invoked as otherwise the random weights/biases are not
            # initialized and loading would fail.
            add_pickle_artifact(_run, model, "model", device=devices.cuda())

        bar.update(step + 1)
    bar.finish()

    return {"model_state": pickle_str(model.state_dict())}
