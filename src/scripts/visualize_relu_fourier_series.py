import numpy as np
import tikzplotlib
import torch
from matplotlib import pyplot as plt
import scipy as sp
import scipy.constants

from rfsf.kernel.initialization.quadrature_fourier_series_initializer import QuadratureFourierSeriesInitializer
from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer
from rfsf.util.tensor_util import periodic


def main():
    half_period = 5
    x = torch.arange(-1.5 * half_period, 1.5 * half_period, 0.01)
    func = periodic(half_period)(QuadratureFourierSeriesInitializer.relu)
    fig, ax = plt.subplots()
    ax.plot(x, func(x), label="Periodic ReLU")
    for num_harmonics in 10 ** np.arange(4, dtype=int):
        init = ReLUFourierSeriesInitializer(num_harmonics, half_period)
        ax.plot(x, init(x) + half_period / 4, label=rf"$K = {num_harmonics}$")
        print(init(torch.tensor([-2.])))
        print(func(torch.tensor([-2.])))
    ax.legend()
    ax.set_title("Periodic ReLU Fourier Series")
    plt.tight_layout()
    fig.savefig("data/temp/figures/relu.pdf")
    fig.show()


if __name__ == "__main__":
    main()
