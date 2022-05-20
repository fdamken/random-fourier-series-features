import numpy as np
import torch
from matplotlib import pyplot as plt

from rfsf.kernel.initialization.relu_fourier_series_initializer import ReLUFourierSeriesInitializer
from rfsf.kernel.rfsf_kernel import RFSFKernel


def main():
    np.random.seed(12345)
    torch.random.manual_seed(12345)

    lengthscale = torch.tensor(1.0)
    x_min, x_max = -5.0, 5.0
    x = torch.arange(x_min, x_max, 0.01).unsqueeze(dim=1)

    num_num_features = 5 - 1
    num_num_harmonics = 7
    num_harmonics_list = 2 ** np.arange(num_num_harmonics, dtype=int)
    num_features_list = 10 ** np.arange(num_num_features, dtype=int)

    matrices = np.empty((num_num_harmonics, num_num_features, len(x), len(x)))
    for i, num_harmonics in enumerate(num_harmonics_list):
        fs_initialization = ReLUFourierSeriesInitializer(num_harmonics, np.pi)
        for j, num_features in enumerate(num_features_list):
            print(f"Computing RFSF for {num_features=} and {num_harmonics=}.")
            rfsf = RFSFKernel(num_features, fs_initialization)
            rfsf.lengthscale = lengthscale
            with torch.no_grad():
                rfsf_mat = rfsf(x, x).detach().cpu().numpy()
            matrices[i, j, :, :] = rfsf_mat
    # vmin, vmax = np.min(matrices), np.max(matrices)

    figsize_multiplier = 2
    fig, axss = plt.subplots(ncols=num_num_harmonics, nrows=num_num_features, figsize=(figsize_multiplier * num_num_harmonics, figsize_multiplier * num_num_features))
    axss = axss.T
    for axs, (i, num_harmonics) in zip(axss, reversed(list(enumerate(num_harmonics_list)))):
        for ax, (j, num_features) in zip(axs, enumerate(num_features_list)):
            ax.imshow(matrices[i, j], extent=[x_min, x_max, x_min, x_max])
            # if i == 0:
            #     ax.set_xlabel(f"$N = {num_features}$")
            # if j == 0:
            #     ax.set_ylabel(f"$K = {num_harmonics}$")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.savefig("data/temp/figures/title2.pdf")
    fig.show()


if __name__ == "__main__":
    main()
