from matplotlib import pyplot as plt

from scripts.plotting.util import savefig
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


@ex.train
def main(__figures_dir: str):
    losses = load_metrics()["loss"]
    steps = losses["steps"]
    values = losses["values"]

    fig, ax = plt.subplots()
    ax.plot(steps, values)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Negative Marginal Log-Likelihood")
    savefig(fig, __figures_dir, "loss").show()


if __name__ == "__main__":
    ex.run()
