import torch

from rfsf.pre_processing.pre_processor import PreProcessor


class Standardization(PreProcessor):
    def __init__(self):
        super().__init__()

    def _fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        # Normalize over the batch dimension (the first axis).
        self.register_buffer("inputs_mean", inputs.mean(dim=0, keepdim=True))
        self.register_buffer("inputs_std", inputs.std(dim=0, keepdim=True))
        self.register_buffer("targets_mean", targets.mean(dim=0, keepdim=True))
        self.register_buffer("targets_std", targets.std(dim=0, keepdim=True))

    def _transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.inputs_mean) / self.inputs_std

    def _inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        return inputs_transformed * self.inputs_std + self.inputs_mean

    def _transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.targets_mean) / self.targets_std

    def _inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        return targets_transformed * self.targets_std + self.targets_mean
