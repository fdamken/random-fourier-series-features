import torch

from rfsf.pre_processing.pre_processor import PreProcessor


class NoOpPreProcessor(PreProcessor):
    def _fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        pass

    def _transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def _transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets

    def _inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        return inputs_transformed

    def _inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        return targets_transformed
