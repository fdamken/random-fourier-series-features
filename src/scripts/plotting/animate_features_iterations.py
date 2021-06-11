from functools import partial
from typing import Tuple

from gpytorch.models import GP
from matplotlib.figure import Figure

from scripts.plotting.common import animate_over_model_states, plot_features
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __y_lim = (-2, 2)
    __max_num_features = 1
    __frame_duration = 50


@ex.main
def main(__figures_dir: str, __y_lim: Tuple[float, float], __max_num_features: int, __frame_duration: int):
    def plot_single(model: GP, title_suffix: str) -> Figure:
        return plot_features(partial(model.cov_module._featurize, identity_randoms=True), __max_num_features, title_suffix=title_suffix)

    animate_over_model_states(load_model(), load_metrics(), load_run(), __figures_dir, "features", plot_single, frame_duration=__frame_duration)


if __name__ == "__main__":
    ex.run()
