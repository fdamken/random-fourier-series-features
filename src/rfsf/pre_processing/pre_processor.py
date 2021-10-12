from abc import ABC, abstractmethod

import torch


class PreProcessor(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._fit_invoked = False

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self._fit(inputs, targets)
        self._fit_invoked = True

    def transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._transform_inputs(inputs)

    def transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._transform_targets(targets)

    def inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_inputs(inputs_transformed)

    def inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_targets(targets_transformed)

    def inverse_transform_covariance_matrix(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_covariance_matrix(covariance_matrix)

    def inverse_transform_target_std_devs(self, target_std_devs_transformed: torch.Tensor) -> torch.Tensor:
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_target_std_devs(target_std_devs_transformed)

    @abstractmethod
    def _fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_covariance_matrix(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_target_std_devs(self, target_std_devs_transformed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
