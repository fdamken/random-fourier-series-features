import sklearn.gaussian_process
import torch

from rfsf.kernel.kernel import Kernel


class SkLearnKernelWrapper(Kernel):
    def __init__(self, kernel: sklearn.gaussian_process.kernels.Kernel):
        self._kernel = kernel

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_np = p.detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        return torch.from_numpy(self._kernel(p_np, q_np).astype(p_np.dtype)).to(p.device)
