import torch

from rfsf.pre_processing.pre_processor import PreProcessor


class Standardization(PreProcessor):
    def _fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        # Normalize over the batch dimension (the first axis).
        self.register_buffer("inputs_mean", inputs.mean(dim=0, keepdim=True))
        inputs_std = inputs.std(dim=0, keepdim=True)
        inputs_std[torch.isclose(inputs_std, torch.tensor(0.0))] = 1.0
        self.register_buffer("inputs_std", inputs_std)
        self.register_buffer("targets_mean", targets.mean(dim=0, keepdim=True))
        targets_std = targets.std(dim=0, keepdim=True)
        targets_std[torch.isclose(targets_std, torch.tensor(0.0))] = 1.0
        self.register_buffer("targets_std", targets_std)

    def _transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.inputs_mean) / self.inputs_std

    def _inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        return inputs_transformed * self.inputs_std + self.inputs_mean

    def _transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.targets_mean) / self.targets_std

    def _inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        return targets_transformed * self.targets_std + self.targets_mean

    def _inverse_transform_covariance_matrix(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        return covariance_matrix @ (torch.eye(covariance_matrix.shape[0]) * self.targets_std ** 2)

    def _inverse_transform_target_std_devs(self, target_std_devs_transformed: torch.Tensor) -> torch.Tensor:
        return target_std_devs_transformed * self.targets_std
