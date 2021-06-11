from gpytorch.models import GP
from matplotlib.figure import Figure

from scripts.plotting.common import animate_over_model_states, plot_covariance
from scripts.plotting.util import AccumulativeNormalization
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __frame_duration = 50
    __normalize = False


@ex.main
def main(__figures_dir: str, __frame_duration: int, __normalize: bool):
    norm = AccumulativeNormalization() if __normalize else None

    def plot_single(model: GP, title_suffix: str) -> Figure:
        return plot_covariance(model, title_suffix=title_suffix, norm=norm)

    animate_over_model_states(load_model(), load_metrics(), load_run(), __figures_dir, "prior-covariance", plot_single, frame_duration=__frame_duration, two_pass=__normalize)


if __name__ == "__main__":
    ex.run()
