from scripts.plotting.common import plot_covariance
from scripts.plotting.util import savefig
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


@ex.main
def main(__figures_dir: str):
    savefig(plot_covariance(load_model()), __figures_dir, "prior-covariance").show()


if __name__ == "__main__":
    ex.run()
