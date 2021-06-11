from typing import Tuple

from gpytorch.models import GP
from matplotlib.figure import Figure

from ingredients import dataset
from scripts.plotting.common import animate_over_model_states, plot_process
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __y_lim = (-2, 2)
    __frame_duration = 50


@ex.main
def main(__figures_dir: str, __y_lim: Tuple[float, float], __frame_duration: int):
    def plot_single(model: GP, title_suffix: str) -> Figure:
        return plot_process(model, 0, dataset.get_title(), y_lim=__y_lim, title_suffix=title_suffix)

    animate_over_model_states(load_model(), load_metrics(), load_run(), __figures_dir, "gp", plot_single, frame_duration=__frame_duration)


if __name__ == "__main__":
    ex.run()
