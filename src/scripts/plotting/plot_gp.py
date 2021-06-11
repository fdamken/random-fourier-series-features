from ingredients import dataset
from scripts.plotting.common import plot_process
from scripts.plotting.util import savefig
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    plot_num_samples = 5


@ex.main
def main(__figures_dir: str, plot_num_samples: int):
    savefig(plot_process(load_model(), dataset.load_data(), plot_num_samples, dataset.get_title()), __figures_dir, "gp").show()


if __name__ == "__main__":
    ex.run()
