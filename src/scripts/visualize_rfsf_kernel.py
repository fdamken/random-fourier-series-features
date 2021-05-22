import numpy as np
import torch
from matplotlib import pyplot as plt

from rfsf.kernel.random_fourier_series_features_kernel import RandomFourierSeriesFeaturesKernel
from rfsf.util import devices


def main():
    np.random.seed(12345)
    torch.random.manual_seed(12345)

    device = devices.cpu()

    length_scale = torch.tensor(1.0)
    x_min, x_max = -5.0, 5.0
    x = torch.arange(x_min, x_max, 0.01, device=device).unsqueeze(dim=1)

    num_num_features = 5
    num_num_harmonics = 7
    figsize_multiplier = 3
    fig, axss = plt.subplots(nrows=num_num_harmonics, ncols=num_num_features, figsize=(figsize_multiplier * num_num_features, figsize_multiplier * num_num_harmonics))
    matrices = np.empty((num_num_harmonics, num_num_features, len(x), len(x)))
    for i, num_harmonics in enumerate(2 ** np.arange(num_num_harmonics, dtype=int)):
        for j, num_features in enumerate(10 ** np.arange(num_num_features, dtype=int)):
            print(f"Computing RFSF for {num_features=} and {num_harmonics=}.")
            rfsf = RandomFourierSeriesFeaturesKernel(1, num_features, num_harmonics, length_scale=length_scale, device=device)
            with torch.no_grad():
                rfsf_mat = rfsf(x, x).detach().cpu().numpy()
            matrices[i, j, :, :] = rfsf_mat
    vmin, vmax = np.min(matrices), np.max(matrices)
    print(f"{vmin=}, {vmax=}")
    for axs, (i, num_harmonics) in zip(axss, reversed(list(enumerate(2 ** np.arange(num_num_harmonics, dtype=int))))):
        for ax, (j, num_features) in zip(axs, enumerate(10 ** np.arange(num_num_features, dtype=int))):
            ax.imshow(matrices[i, j], extent=[x_min, x_max, x_min, x_max])
            if i == 0:
                ax.set_xlabel(f"$N = {num_features}$")
            if j == 0:
                ax.set_ylabel(f"$K = {num_harmonics}$")
            ax.set_xticks([])
            ax.set_yticks([])
            scale = (vmax - vmin) / (np.max(matrices[i, j]) - np.min(matrices[i, j]))
            ax.set_title(f"Scale: {scale:.2f}")
    plt.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()
