from functools import partial
from typing import Tuple

from gpytorch.models import GP
from matplotlib.figure import Figure

from rfsf.pre_processing.pre_processor import PreProcessor
from scripts.plotting.common import animate_over_model_states, plot_features
from scripts.plotting.util import show_debug_info
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model, load_pre_processor = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __y_lim = None
    __max_num_features = 1
    __frame_duration = 50


@ex.main
def main(__figures_dir: str, __experiment_dir: str, __y_lim: Tuple[float, float], __max_num_features: int, __frame_duration: int):
    def plot_single(pre_processor: PreProcessor, model: GP, title_suffix: str) -> Figure:
        return show_debug_info(
            plot_features(partial(model.cov_module._featurize, identity_randoms=True), __max_num_features, y_lim=__y_lim, title_suffix=title_suffix),
            load_run(),
            __experiment_dir,
        )

    animate_over_model_states(load_pre_processor(), load_model(), load_metrics(), load_run(), __figures_dir, "features", plot_single, frame_duration=__frame_duration)


if __name__ == "__main__":
    ex.run()
