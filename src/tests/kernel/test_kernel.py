import pytest
import torch

from rfsf.kernel.kernel import Kernel


class MockKernel(Kernel):
    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        pass


@pytest.mark.parametrize("param_a_grad", [True, False])
@pytest.mark.parametrize("param_b_grad", [True, False])
@pytest.mark.parametrize("param_c_grad", [True, False])
def test_register_parameters(param_a_grad, param_b_grad, param_c_grad):
    param_a = torch.tensor(0.0, requires_grad=param_a_grad)
    param_b = torch.tensor(0.0, requires_grad=param_b_grad)
    param_c = torch.tensor(0.0, requires_grad=param_c_grad)
    expected_parameters = []
    if param_a_grad:
        expected_parameters.append(param_a)
    if param_b_grad:
        expected_parameters.append(param_b)
    if param_c_grad:
        expected_parameters.append(param_c)
    kernel = MockKernel()
    kernel.register_parameters(param_a, param_b, param_c)
    assert len(kernel.parameters) == param_a_grad + param_b_grad + param_c_grad
    assert kernel.parameters == expected_parameters
