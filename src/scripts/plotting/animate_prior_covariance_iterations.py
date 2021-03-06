from gpytorch.models import GP
from matplotlib.figure import Figure

from rfsf.pre_processing.pre_processor import PreProcessor
from scripts.plotting.common import animate_over_model_states, plot_epistemic_covariance
from scripts.plotting.util import AccumulativeNormalization, show_debug_info
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __normalize = False
    __frame_duration = 50


@ex.main
def main(__figures_dir: str, __experiment_dir: str, __frame_duration: int, __normalize: bool):
    norm = AccumulativeNormalization() if __normalize else None

    def plot_single(pre_processor: PreProcessor, model: GP, title_suffix: str) -> Figure:
        return show_debug_info(
            plot_epistemic_covariance(pre_processor, model, title_suffix=title_suffix, norm=norm),
            load_run(),
            __experiment_dir,
        )

    animate_over_model_states(load_pre_processor(), iterate_models(), __figures_dir, "prior-covariance", plot_single, frame_duration=__frame_duration, two_pass=__normalize)


if __name__ == "__main__":
    ex.run()
