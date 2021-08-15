import torch
from sklearn.decomposition import PCA

from rfsf.pre_processing.pre_processor import PreProcessor
from rfsf.util.tensor_util import process_as_numpy_array


class PCAWhitening(PreProcessor):
    def _fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        pca = PCA(whiten=True)
        pca.fit(inputs.detach().cpu().numpy())
        self.register_buffer("inputs_pca_components", torch.from_numpy(pca.components_))
        self.register_buffer("inputs_pca_mean", torch.from_numpy(pca.mean_))
        self.register_buffer("inputs_pca_explained_variance", torch.from_numpy(pca.explained_variance_))

        targets_mean = targets.mean(dim=0, keepdim=True)
        targets_std = targets.std(dim=0, keepdim=True)
        targets_std[torch.isclose(targets_std, torch.tensor(0.0))] = 1.0
        self.register_buffer("targets_mean", targets_mean)
        self.register_buffer("targets_std", targets_std)

    def _transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return process_as_numpy_array(inputs, self._pca.transform)

    def _inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        return process_as_numpy_array(inputs_transformed, self._pca.inverse_transform)

    def _transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.targets_mean) / self.targets_std

    def _inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        return targets_transformed * self.targets_std + self.targets_mean

    def _inverse_transform_target_std_devs(self, target_std_devs_transformed: torch.Tensor) -> torch.Tensor:
        return target_std_devs_transformed * self.targets_std

    @property
    def _pca(self) -> PCA:
        pca = PCA(whiten=True)
        pca.components_ = self.inputs_pca_components.detach().cpu().numpy()
        pca.mean_ = self.inputs_pca_mean.detach().cpu().numpy()
        pca.explained_variance_ = self.inputs_pca_explained_variance.detach().cpu().numpy()
        return pca
