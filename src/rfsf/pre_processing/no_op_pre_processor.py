import torch

from rfsf.pre_processing.pre_processor import PreProcessor


class NoOpPreProcessor(PreProcessor):
    """Pre-processor that does nothing and returns the inputs/targets as is."""

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

    def _inverse_transform_target_std_devs(self, target_std_devs_transformed: torch.Tensor) -> torch.Tensor:
        return target_std_devs_transformed
