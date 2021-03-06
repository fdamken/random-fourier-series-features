import numpy as np
from matplotlib import pyplot as plt

from scripts.plotting.util import savefig, show_debug_info
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor = load_experiment()


@ex.main
def main(__figures_dir: str, __experiment_dir: str):
    train_losses = load_metrics()["loss"]
    train_steps = train_losses["steps"]
    train_values = train_losses["values"]

    fig, ax = plt.subplots()
    ax.plot(train_steps, -np.asarray(train_values), label="Train")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Likelihood")
    ax.set_title("Marginal Log-Likelihood")
    ax.legend()
    savefig(show_debug_info(fig, load_run(), __experiment_dir), __figures_dir, "loss").show()


if __name__ == "__main__":
    ex.run()
