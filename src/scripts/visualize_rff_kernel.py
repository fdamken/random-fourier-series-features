import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF

from rfsf.kernel.random_fourier_features_kernel import RandomFourierFeaturesKernel
from rfsf.kernel.sklearn_kernel_wrapper import SkLearnKernelWrapper


def main():
    np.random.seed(12345)
    torch.random.manual_seed(12345)

    length_scale = 1.0
    rbf = SkLearnKernelWrapper(RBF(length_scale=length_scale))

    x = torch.arange(-5.0, 5.0, 0.01).unsqueeze(dim=1)

    rbf_mat = rbf(x, x).numpy()

    fig, axss = plt.subplots(nrows=2, ncols=7, figsize=(21, 6))
    (ax11, ax12), *axs = axss.T
    ax11.imshow(rbf_mat, extent=[x.min(), x.max(), x.min(), x.max()])
    ax12.imshow(np.isclose(rbf_mat, rbf_mat), extent=[x.min(), x.max(), x.min(), x.max()], vmin=0, vmax=1)
    ax11.set_title("Exact RBF Kernel")
    for i, (ax21, ax22) in enumerate(axs):
        num_features = 10 ** i
        rff = RandomFourierFeaturesKernel(1, num_features, length_scale=torch.tensor(length_scale))
        rff_mat = rff(x, x).numpy()
        ax21.imshow(rff_mat, extent=[x.min(), x.max(), x.min(), x.max()])
        ax22.imshow(np.isclose(rff_mat, rbf_mat, atol=0.01), extent=[x.min(), x.max(), x.min(), x.max()], vmin=0, vmax=1)
        ax21.set_title(f"RFF Kernel, {num_features} Features")
    plt.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()
