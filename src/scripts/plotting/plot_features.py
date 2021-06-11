from scripts.plotting.common import plot_features
from scripts.plotting.util import savefig
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __num_features = 10


@ex.main
def main(__figures_dir: str, __num_features: int):
    # noinspection PyProtectedMember
    savefig(plot_features(load_model().cov_module._featurize, __num_features), __figures_dir, "features").show()


if __name__ == "__main__":
    ex.run()
