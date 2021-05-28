import numpy as np
import torch
from matplotlib import pyplot as plt

from rfsf.kernel.initialization.quadrature_rourier_series_initializer import QuadratureFourierSeriesInitializer
from rfsf.util.tensor_util import periodic


def main(func, name):
    half_period = 5
    x = torch.arange(-1.5 * half_period, 1.5 * half_period, 0.01)
    expected = torch.from_numpy(periodic(half_period)(func)(x.detach().cpu().numpy()))
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for ax, num_harmonics in zip(axs.ravel(), 10 ** np.arange(4, dtype=int)):
        init = QuadratureFourierSeriesInitializer(func, num_harmonics, half_period, torch_dtype=torch.float)
        actual = init(x)
        ax.plot(x, expected, label=name)
        ax.plot(x, actual, label=f"Numerical {name} Series")
        ax.set_title(f"{num_harmonics} Harmonics")
        ax.legend()
        if torch.linalg.norm(actual - expected) < 0.1:
            break
    plt.tight_layout()
    fig.show()


if __name__ == "__main__":
    print("Calculating Erf Fourier series.")
    main(QuadratureFourierSeriesInitializer.erf, "Erf")

    print("Calculating ReLU Fourier series.")
    main(QuadratureFourierSeriesInitializer.relu, "ReLU")

    print("Calculating Tanh Fourier series.")
    main(QuadratureFourierSeriesInitializer.tanh, "Tanh")
