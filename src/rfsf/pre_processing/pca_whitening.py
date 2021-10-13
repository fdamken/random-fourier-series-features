import torch
from sklearn.decomposition import PCA

from rfsf.pre_processing.standardization import Standardization
from rfsf.util.tensor_util import process_as_numpy_array


class PCAInputWhitening(Standardization):
    """
    Pre-processor for whitening the inputs using a PCA transformation. The targets are still standardized using
    z-scores.
    """

    def __init__(self):
        super().__init__(disable_input_standardization=True)

    def _fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        super()._fit(inputs, targets)
        pca = PCA(whiten=True)
        pca.fit(inputs.detach().cpu().numpy())
        self.register_buffer("inputs_pca_components", torch.from_numpy(pca.components_))
        self.register_buffer("inputs_pca_mean", torch.from_numpy(pca.mean_))
        self.register_buffer("inputs_pca_explained_variance", torch.from_numpy(pca.explained_variance_))

    def _transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return process_as_numpy_array(inputs, self._pca.transform)

    def _inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        return process_as_numpy_array(inputs_transformed, self._pca.inverse_transform)

    @property
    def _pca(self) -> PCA:
        pca = PCA(whiten=True)
        pca.components_ = self.inputs_pca_components.detach().cpu().numpy()
        pca.mean_ = self.inputs_pca_mean.detach().cpu().numpy()
        pca.explained_variance_ = self.inputs_pca_explained_variance.detach().cpu().numpy()
        return pca
