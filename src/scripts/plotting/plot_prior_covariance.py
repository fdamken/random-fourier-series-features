from scripts.plotting.common import plot_epistemic_covariance
from scripts.plotting.util import savefig, show_debug_info
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor = load_experiment()


@ex.main
def main(__figures_dir: str, __experiment_dir: str):
    savefig(show_debug_info(plot_epistemic_covariance(load_pre_processor(), load_model()), load_run(), __experiment_dir), __figures_dir, "prior-covariance").show()


if __name__ == "__main__":
    ex.run()
