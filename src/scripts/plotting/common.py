from typing import Callable, Iterable, Optional, Tuple

import gpytorch
import torch
from gpytorch.models import GP
from matplotlib import colors, cycler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

from ingredients import dataset
from rfsf.pre_processing.pre_processor import PreProcessor
from rfsf.util import devices
from rfsf.util.tensor_util import to_numpy
from scripts.plotting.util import savefig


sample_color_cycler = cycler(color=["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])


def animate_over_model_states(
    pre_processor: PreProcessor,
    models: Iterable[Tuple[Optional[int], GP]],
    figures_dir: str,
    filename: str,
    plot_single: Callable[[PreProcessor, GP, str], Figure],
    *,
    two_pass: bool = False,
    frame_duration: int = 50,
):
    filenames = []
    passes = [True]
    if two_pass:
        passes = [False] + passes
    for i, save_fig in enumerate(passes):
        print(f"Plotting steps (pass {i + 1}, {'' if save_fig else 'not '}saving).")
        filenames = []
        for step, model in tqdm(models):
            model.eval()
            if step is None:
                frame_filename = "final"
            else:
                frame_filename = f"{step:010d}"
            fig = plot_single(pre_processor, model, "; " + ("Final" if step is None else f"Step {step}"))
            if save_fig:
                savefig(fig, figures_dir, frame_filename, formats=["png"])
                filenames.append(frame_filename)
            plt.close(fig)

    print("Plotting finished. Generating GIF.")
    frames = [Image.open(f"{figures_dir}/{filename}.png") for filename in tqdm(filenames)]
    frames[0].save(f"{figures_dir}/{filename}.gif", save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)


def plot_process(
    pre_processor: PreProcessor,
    model: GP,
    num_samples: int,
    title: str,
    *,
    title_suffix: str = "",
    legend: bool = True,
    legend_loc: Optional[str] = "lower left",
    y_lim: Optional[Tuple[float, float]] = None,
    fig_ax: Optional[Tuple[Figure, Axes]] = None,
) -> Figure:
    (train_x, train_y), (test_x, test_y) = dataset.load_data(device=devices.cuda())
    pre_processor.to(devices.cuda())
    model.to(devices.cuda())
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x_transformed = pre_processor.transform_inputs(test_x)
        pred = model(test_x_transformed)
    fig, ax = plt.subplots() if fig_ax is None else fig_ax
    pred_mean = pre_processor.inverse_transform_targets(pred.mean)
    # TODO: Is this inverse transformation correct? It does not seem so, but the plots look good…
    pred_conf = torch.tensor(2) * pre_processor.inverse_transform_target_std_devs(pred.stddev)
    lower, upper = pred_mean - pred_conf, pred_mean + pred_conf
    # Make one-dimensional for plotting.
    train_x, test_x = train_x.squeeze(), test_x.squeeze()
    ax.scatter(to_numpy(train_x), to_numpy(train_y), color="black", marker="*", s=100, label="Observed Data", zorder=3)
    ax.plot(to_numpy(test_x), to_numpy(test_y), color="black", label="True Func.", zorder=0)
    for _, c in zip(range(num_samples), sample_color_cycler):
        ax.plot(to_numpy(test_x), to_numpy(pre_processor.inverse_transform_targets(pred.sample())), color=c["color"], alpha=0.3, zorder=2)
    ax.plot(to_numpy(test_x), to_numpy(pred_mean), color="tab:blue", label="Mean", zorder=3)
    ax.fill_between(to_numpy(test_x), to_numpy(lower), to_numpy(upper), color="tab:blue", alpha=0.2, label=r"Confidence", zorder=1)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_title(title + title_suffix)
    if legend:
        ax.legend(loc=legend_loc)
    return fig


def plot_features(
    featurize: Callable[[torch.Tensor], torch.Tensor],
    max_num_features,
    *,
    title_suffix: str = "",
    y_lim: Optional[Tuple[float, float]] = None,
    fig_ax: Optional[Tuple[Figure, Axes]] = None,
) -> Figure:
    _, (test_inputs, _) = dataset.load_data()
    test_inputs = torch.arange(2 * test_inputs.min(), 2 * test_inputs.max(), 0.01)
    fig, ax = plt.subplots() if fig_ax is None else fig_ax
    features = to_numpy(featurize(test_inputs.unsqueeze(-1)))
    ax.plot(test_inputs, features[:, :max_num_features])
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\phi_i(x)$")
    ax.set_title(f"Features{title_suffix}")
    return fig


def plot_covariance(
    pre_processor: PreProcessor,
    model: GP,
    *,
    title_suffix: str = "",
    posterior: bool = False,
    show_cbar: bool = True,
    norm: Optional[colors.Normalize] = None,
    fig_ax: Optional[Tuple[Figure, Axes]] = None,
) -> Figure:
    inputs = torch.arange(-5, +5, 0.01)
    inputs_transformed = pre_processor.transform_inputs(inputs)
    targets = inputs
    if posterior:
        model.eval()
    else:
        # Hack the test data as training data into the model to compute the kernel and not the posterior covariance.
        model.set_train_data(inputs_transformed, pre_processor.transform_targets(targets), strict=False)
        model.train()
    x_min, x_max = inputs.min(), inputs.max()
    fig, ax = plt.subplots() if fig_ax is None else fig_ax
    mappable = ax.imshow(to_numpy(model(inputs_transformed).covariance_matrix), extent=[x_min, x_max, x_min, x_max], norm=norm)
    if show_cbar:
        fig.colorbar(mappable)
    ax.set_aspect(1.0)
    ax.set_title(f"{'Posterior' if posterior else 'Prior'} Covariance{title_suffix}")
    return fig
