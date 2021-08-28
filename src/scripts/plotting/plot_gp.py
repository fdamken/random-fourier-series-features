from ingredients import dataset
from scripts.plotting.common import plot_process
from scripts.plotting.util import savefig, show_debug_info
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __num_samples = 5


@ex.main
def main(__figures_dir: str, __experiment_dir: str, __num_samples: int):
    savefig(
        show_debug_info(
            plot_process(load_pre_processor(), load_model(), __num_samples, dataset.get_title()),
            load_run(),
            __experiment_dir,
        ),
        __figures_dir,
        "gp",
    ).show()


if __name__ == "__main__":
    ex.run()
