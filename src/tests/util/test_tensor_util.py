import numpy as np
import pytest
import torch

from rfsf.util.tensor_util import is_numpy, is_torch, periodic


def test_is_numpy():
    assert is_numpy(np.array(0.0))
    assert is_numpy(torch.tensor(0.0).numpy())
    assert not is_numpy(torch.tensor(0.0))


def test_is_torch():
    assert is_torch(torch.FloatTensor([0.0]))
    assert is_torch(torch.IntTensor([0]))
    assert is_torch(torch.tensor(0))
    assert is_torch(torch.from_numpy(np.array(0.0)))
    assert not is_torch(np.array(0.0))


@pytest.mark.parametrize("trig", [np.sin, np.cos, np.tan])
def test_periodic(trig):
    truncated_trig = lambda x: trig(np.clip(x, -np.pi, np.pi))
    func = periodic(np.pi)(truncated_trig)
    x = np.arange(-10, 10, 0.01)
    expected = trig(x)
    assert np.allclose(func(x), expected)
