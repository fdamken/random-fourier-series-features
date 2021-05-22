import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF

from rfsf.kernel.random_fourier_features_kernel import RandomFourierFeaturesKernel
from rfsf.kernel.sklearn_kernel_wrapper import SkLearnKernelWrapper


def main():
    np.random.seed(12345)
    torch.random.manual_seed(12345)

    length_scale = torch.tensor(1.0)
    rbf = SkLearnKernelWrapper(RBF(length_scale=length_scale.item()))

    x_min, x_max = -5.0, 5.0
    x = torch.arange(x_min, x_max, 0.01).unsqueeze(dim=1)

    rbf_mat = rbf(x, x).numpy()

    num_num_features = 7
    num_features_list = 10 ** np.arange(num_num_features, dtype=int)

    matrices = np.empty((num_num_features + 1, len(x), len(x)))
    matrices[0, :, :] = rbf_mat
    for i, num_features in enumerate(num_features_list):
        print(f"Computing RFF for {num_features=}.")
        rfsf = RandomFourierFeaturesKernel(1, num_features, length_scale=length_scale)
        with torch.no_grad():
            rfsf_mat = rfsf(x, x).detach().cpu().numpy()
        matrices[i + 1, :, :] = rfsf_mat
    # mmax = max(np.max(matrices), np.max(rbf_mat))

    figsize_multiplier = 2
    fig, axss = plt.subplots(nrows=2, ncols=num_num_features, figsize=(figsize_multiplier * (num_num_features + 1), 1 + figsize_multiplier * 2))
    for (ax1, ax2), (i, num_features) in zip(axss.T, enumerate(num_features_list)):
        rff_mat = matrices[i]
        ax1.imshow(rff_mat, extent=[x_min, x_max, x_min, x_max])
        ax2.imshow(np.isclose(rff_mat, rbf_mat, atol=0.01), extent=[x.min(), x.max(), x.min(), x.max()], vmin=0, vmax=1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        if i == 0:
            ax1.set_title("Exact RBF Kernel")
        else:
            ax1.set_title(f"$N = {num_features}$")
    plt.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()
