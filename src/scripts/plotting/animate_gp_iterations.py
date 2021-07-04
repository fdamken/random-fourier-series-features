from typing import Tuple

from gpytorch.models import GP
from matplotlib.figure import Figure

from ingredients import dataset
from rfsf.kernel.rfsf_kernel import RFSFKernel
from rfsf.pre_processing.pre_processor import PreProcessor
from scripts.plotting.common import animate_over_model_states, plot_process
from scripts.plotting.util import show_debug_info
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model, load_pre_processor = load_experiment()


class Value:
    def __init__(self, value):
        self.value = value


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __y_lim = (-0.5, 1.5)
    __frame_duration = 50


@ex.main
def main(__figures_dir: str, __experiment_dir: str, __y_lim: Tuple[float, float], __frame_duration: int):
    weights_and_phases = Value(None)

    def plot_single(pre_processor: PreProcessor, model: GP, title_suffix: str) -> Figure:
        if hasattr(model, "cov_module") and isinstance(model.cov_module, RFSFKernel):
            if weights_and_phases.value is None:
                weights_and_phases.value = model.cov_module.get_weights_and_phases(None)
            model.cov_module.set_weights_and_phases(*weights_and_phases.value)
        return show_debug_info(
            plot_process(pre_processor, model, 0, dataset.get_title(), y_lim=__y_lim, title_suffix=title_suffix),
            load_run(),
            __experiment_dir,
        )

    animate_over_model_states(load_pre_processor(), load_model(), load_metrics(), load_run(), __figures_dir, "gp", plot_single, frame_duration=__frame_duration)


if __name__ == "__main__":
    ex.run()
