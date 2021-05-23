import pytest
import torch

from rfsf.util import devices


@pytest.fixture(scope="function")
def make_cuda_unavailable():
    orig_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    yield None
    torch.cuda.is_available = orig_cuda_is_available


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on the test platform")
def test_cuda():
    expected = torch.tensor(0.0).to(device="cuda").device
    assert devices.cuda() == expected


def test_cuda_not_available(make_cuda_unavailable):
    with pytest.raises(AssertionError) as exc_info:
        devices.cuda()
    assert exc_info.value.args == ("CUDA is not available",)


def test_cpu():
    expected = torch.tensor(0.0).to(device="cpu").device
    assert devices.cpu() == expected
