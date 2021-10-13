from abc import ABC, abstractmethod

import torch


class PreProcessor(ABC, torch.nn.Module):
    """
    Torch module that acts as a pre-processor for a regression problem. This is an abstract base class, i.e., some
    methods have to be overwritten by subclasses.

    The inputs and targets (which are assumed to be scalars) are transformed independently and the pre-processor has to
    be fit to training data using the :py:meth:`.fit` method before using the transformation methods.
    """

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._fit_invoked = False

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Fits the pre-processor's internal state to the given inputs and targets."""
        self._fit(inputs, targets)
        self._fit_invoked = True

    def transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transforms the given inputs to be used by a subsequent learner."""
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._transform_inputs(inputs)

    def transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Transforms the given targets to be used by a subsequent learner."""
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._transform_targets(targets)

    def inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        """Inversely transforms the given inputs."""
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_inputs(inputs_transformed)

    def inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        """Inversely transforms the given targets, e.g., to transform a mean prediction."""
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_targets(targets_transformed)

    def inverse_transform_target_cov(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Inversely transforms the given target covariance, i.e., the predictive covariance produced by a probabilistic
        model. Can, for example, be used to construct the predictive distribution in the original space.

        :param covariance_matrix: covariance matrix of a predictive distribution in the transformed space
        :return: covariance matrix of a predictive distribution in the original space
        """
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_target_cov(covariance_matrix)

    def inverse_transform_target_std_devs(self, target_std_devs_transformed: torch.Tensor) -> torch.Tensor:
        """
        Computes the standard deviations in the original space from the predictive standard deviations in the
        transformed space.

        .. seealso::
            Corresponds to the square-root of the diagonal elements of :py:meth:`.inverse_transform_target_cov`.

        :param target_std_devs_transformed: standard deviations of a predictive distribution in the transformed space
        :return: standard deviations of the a predictive distribution in the original space
        """
        assert self._fit_invoked, "fit() has to be invoked before transforming data"
        return self._inverse_transform_target_std_devs(target_std_devs_transformed)

    @abstractmethod
    def _fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Actual implementation of :py:meth:`.fit` to be overwritten by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def _transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Actual implementation of :py:meth:`.transform_inputs` to be overwritten by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def _transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Actual implementation of :py:meth:`.transform_targets` to be overwritten by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_inputs(self, inputs_transformed: torch.Tensor) -> torch.Tensor:
        """Actual implementation of :py:meth:`.inverse_transform_inputs` to be overwritten by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_targets(self, targets_transformed: torch.Tensor) -> torch.Tensor:
        """Actual implementation of :py:meth:`.inverse_transform_targets` to be overwritten by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_target_cov(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        """Actual implementation of :py:meth:`.inverse_transform_target_cov` to be overwritten by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def _inverse_transform_target_std_devs(self, target_std_devs_transformed: torch.Tensor) -> torch.Tensor:
        """Actual implementation of :py:meth:`.inverse_transform_target_std_devs` to be overwritten by subclasses."""
        raise NotImplementedError()
