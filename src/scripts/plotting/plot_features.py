from functools import partial

from scripts.plotting.common import plot_features
from scripts.plotting.util import savefig
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


# noinspection PyUnusedLocal
@ex.config
def default_config():
    __max_num_features = 5


@ex.main
def main(__figures_dir: str, __max_num_features: int):
    # noinspection PyProtectedMember
    savefig(plot_features(partial(load_model().cov_module._featurize, identity_randoms=False), __max_num_features), __figures_dir, "features").show()


if __name__ == "__main__":
    ex.run()
