import pytest
import torch
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, ExpSineSquared, Matern, RationalQuadratic, WhiteKernel

from rfsf.kernel.sklearn_kernel_wrapper import SkLearnKernelWrapper


@pytest.mark.parametrize("reference", [ConstantKernel(), DotProduct(), ExpSineSquared(), Matern(), RBF(), RationalQuadratic(), WhiteKernel()])
@pytest.mark.parametrize("samples_p", [1, 2, 3, 5])
@pytest.mark.parametrize("samples_q", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_forward(reference, samples_p, samples_q, dim):
    wrapper = SkLearnKernelWrapper(reference)
    p = torch.rand(size=(samples_p, dim))
    q = torch.rand(size=(samples_q, dim))
    assert torch.allclose(wrapper(p, q), torch.from_numpy(reference(p.numpy(), q.numpy()).astype(p.numpy().dtype)))


def test_dtype():
    x = torch.rand(size=(10, 5))
    kernel = SkLearnKernelWrapper(RBF())
    assert kernel(x, x).dtype == x.dtype
