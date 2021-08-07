import pickle
from typing import Callable, Final, List, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.models import GP
from tqdm import tqdm

from ingredients import dataset
from rfsf.util import devices
from scripts.plotting.util import savefig, show_debug_info
from scripts.util.sacred_util import load_experiment


amplitudes_param_name: Final[str] = "Amplitudes"
phases_param_name: Final[str] = "Phases"

ex, load_config, load_metrics, load_run, load_model, load_pre_processor = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __use_test_data = False
    __amplitudes_limits = (0, 4)
    __phases_limits = (-2 * np.pi, 2 * np.pi)
    __plot_amplitudes = True
    __plot_phases = True
    __num_evaluations = 2
    __save_figure = True
    __show_figure = False


def compute_loss(mll: ExactMarginalLogLikelihood, model: GP, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        out = model(inputs)
        return mll(out, targets).item()


@ex.capture
def calculate_losses_grid(
    grid_axis_ticks: List[Tuple[int, int, str, Tuple[float, float], np.ndarray]],
    params: List[torch.nn.Parameter],
    evaluate: Callable[[], float],
    __num_evaluations: int,
) -> List[List[Tuple[str, str, Tuple[float, float], Tuple[float, float], float, float, np.ndarray]]]:
    losses_grid = []
    number_of_non_duplicate_cells = len(grid_axis_ticks) * (len(grid_axis_ticks) - 1) // 2  # Gaussian sum rule.
    with tqdm(total=number_of_non_duplicate_cells * __num_evaluations ** 2, desc="Calculating") as pbar:
        for x, (x_tick_param, x_tick_param_idx, x_tick_param_name, x_tick_param_limits, x_tick_param_values) in enumerate(grid_axis_ticks):
            sub_losses_grid = []
            for y, (y_tick_param, y_tick_param_idx, y_tick_param_name, y_tick_param_limits, y_tick_param_values) in enumerate(grid_axis_ticks):
                x_tick_orig_param_value = params[x_tick_param][x_tick_param_idx].item()
                y_tick_orig_param_value = params[y_tick_param][y_tick_param_idx].item()
                if x == y:
                    losses = np.zeros((len(x_tick_param_values), len(y_tick_param_values)))
                elif x > y:
                    losses = losses_grid[y][x][-1].T
                else:
                    losses = []
                    for x_tick_param_value in x_tick_param_values:
                        sub_losses = []
                        params[x_tick_param][x_tick_param_idx] = x_tick_param_value.item()
                        for y_tick_param_value in y_tick_param_values:
                            params[y_tick_param][y_tick_param_idx] = y_tick_param_value.item()
                            sub_losses.append(evaluate())
                            pbar.update()
                        losses.append(sub_losses)
                params[x_tick_param][x_tick_param_idx] = x_tick_orig_param_value
                params[y_tick_param][y_tick_param_idx] = y_tick_orig_param_value
                sub_losses_grid.append(
                    (x_tick_param_name, y_tick_param_name, x_tick_param_limits, y_tick_param_limits, x_tick_orig_param_value, y_tick_orig_param_value, np.asarray(losses))
                )
            losses_grid.append(sub_losses_grid)
    return losses_grid


def assert_is_symmetric(grid_size: int, losses_grid: List[List[Tuple[str, str, Tuple[float, float], Tuple[float, float], float, float, np.ndarray]]]) -> None:
    asymmetric_indices = []
    for x in range(grid_size):
        for y in range(x + 1, grid_size):
            if not np.allclose(losses_grid[x][y][-1], losses_grid[y][x][-1].T):
                asymmetric_indices.append((x, y))
    assert len(asymmetric_indices) <= 0, f"Losses grid is not symmetric for {sorted(asymmetric_indices)}!"


@ex.capture
def compute_data(
    __use_test_data: bool,
    __amplitudes_limits: Tuple[float, float],
    __phases_limits: Tuple[float, float],
    __plot_amplitudes: bool,
    __plot_phases: bool,
    __num_evaluations: int,
) -> List[List[Tuple[str, str, Tuple[float, float], Tuple[float, float], float, float, np.ndarray]]]:
    model, pre_processor = load_model(), load_pre_processor()
    inputs, targets = dataset.load_data(device=devices.cuda())[1 if __use_test_data else 0]
    pre_processor.to(devices.cuda())
    model.to(devices.cuda())
    model.train()
    model.likelihood.train()
    inputs = pre_processor.transform_inputs(inputs)
    targets = pre_processor.transform_targets(targets)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    params = []
    params_names = []
    params_limits = []
    params_value_transformations = []
    if __plot_amplitudes:
        params.append(model.cov_module.amplitudes_sqrt)
        params_names.append(amplitudes_param_name)
        params_limits.append(__amplitudes_limits)
        params_value_transformations.append(np.sqrt)
    if __plot_phases:
        params.append(model.cov_module.phases)
        params_names.append(phases_param_name)
        params_limits.append(__phases_limits)
        params_value_transformations.append(lambda x: x)
    assert len(params) > 0, "At least one of __plot_amplitudes and __plot_phases must be set to True!"
    for param in params:
        param.requires_grad = False

    grid_axis_ticks = []
    for i, (param, param_name, param_limits, params_value_transformation) in enumerate(zip(params, params_names, params_limits, params_value_transformations)):
        values = params_value_transformation(np.linspace(*param_limits, __num_evaluations))
        grid_axis_ticks += [(i, j, f"{param_name} [{j}]", param_limits, values) for j in range(len(param))]

    grid_size = len(grid_axis_ticks)
    losses_grid = calculate_losses_grid(grid_axis_ticks, params, evaluate=lambda: compute_loss(mll, model, inputs, targets))
    assert_is_symmetric(grid_size, losses_grid)
    return losses_grid


def plot_losses_grid(losses_grid: List[List[Tuple[str, str, Tuple[float, float], Tuple[float, float], float, float, np.ndarray]]]) -> plt.Figure:
    grid_size = len(losses_grid)
    fig, axss = plt.subplots(ncols=grid_size, nrows=grid_size, figsize=(grid_size * 5, grid_size * 5))
    with tqdm(total=2 * grid_size ** 2, desc="   Plotting") as pbar:
        for x, (axs, sub_losses_grid) in enumerate(zip(axss.T, losses_grid)):
            for y, (ax, (x_tick_param_name, y_tick_param_name, x_tick_param_limits, y_tick_param_limits, x_tick_orig_param_value, y_tick_orig_param_value, losses)) in enumerate(
                zip(axs, sub_losses_grid)
            ):
                ax.imshow(losses.T, extent=[*x_tick_param_limits, *y_tick_param_limits], aspect="auto", cmap="binary" if x == y else "winter")
                if amplitudes_param_name in x_tick_param_name:
                    x_tick_orig_param_value = x_tick_orig_param_value ** 2
                if amplitudes_param_name in y_tick_param_name:
                    y_tick_orig_param_value = y_tick_orig_param_value ** 2
                ax.axvline(x_tick_orig_param_value, color="black", linewidth=4)
                ax.axhline(y_tick_orig_param_value, color="black", linewidth=4)
                if y == 0:
                    ax.set_xlabel(x_tick_param_name)
                if x == 0:
                    ax.set_ylabel(y_tick_param_name)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.xaxis.set_label_position("top")
                pbar.update()
        plt.tight_layout()
    return fig


@ex.capture
def plot_data(
    losses_grid: List[List[Tuple[str, str, Tuple[float, float], Tuple[float, float], float, float, np.ndarray]]],
    __figures_dir: str,
    __experiment_dir: str,
    __save_figure: bool,
    __show_figure: bool,
) -> None:
    fig = plot_losses_grid(losses_grid)
    print("Adding debug information.")
    fig = show_debug_info(fig, load_run(), __experiment_dir)
    if __save_figure:
        print("Saving figure.")
        fig = savefig(fig, __figures_dir, "loss_landscape", formats=["png"])
    if __show_figure:
        print("Showing figure.")
        fig.show()


@ex.main
def main(
    __eval_dir: str,
    __load_dumped_eval: bool,
):
    dump_file = f"{__eval_dir}/loss_landscape.pkl"
    if __load_dumped_eval:
        print("Loading dumped evaluation.")
        with open(dump_file, "rb") as f:
            losses_grid = pickle.load(f)
    else:
        losses_grid = compute_data()
        # print("Dumping evaluation.")
        # with open(dump_file, "wb") as f:
        #     pickle.dump(losses_grid, f)
    plot_data(losses_grid)


if __name__ == "__main__":
    ex.run()
