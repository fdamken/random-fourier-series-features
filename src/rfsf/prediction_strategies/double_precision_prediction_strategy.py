import gpytorch.lazy
import torch
from gpytorch import delazify, lazify
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy


class DoublePrecisionPredictionStrategy(DefaultPredictionStrategy):
    """Custom prediction strategy that uses double precision floating points for computing the posterior covariance."""

    def exact_predictive_covar(self, test_test_covar: gpytorch.lazy.LazyTensor, test_train_covar: gpytorch.lazy.LazyTensor) -> gpytorch.lazy.LazyTensor:
        """
        Computes the posterior predictive covariance of a GP. Compared to the superclass implementation, this
        implementation uses double floating point precision for the matrix inversion. This can prevent non-positive
        definite or even asymmetric covariance matrices.

        :param test_train_covar: covariance matrix between test and train input
        :param test_test_covar: covariance matrix between test inputs
        :return: lazy tensor representing the predictive posterior covariance of the test points
        """

        # These cases are handled by the super class.
        if gpytorch.settings.fast_pred_var.on() or gpytorch.settings.skip_posterior_variances.on():
            return super().exact_predictive_covar(test_test_covar, test_train_covar)

        dist = self.train_prior_dist.__class__(torch.zeros_like(self.train_prior_dist.mean), self.train_prior_dist.lazy_covariance_matrix)
        train_train_covar = self.likelihood(dist, self.train_inputs).covariance_matrix

        test_test_covar_d = delazify(test_test_covar).detach().double()
        test_train_covar_d = delazify(test_train_covar).detach().double()
        train_train_covar_d = delazify(train_train_covar).detach().double()
        covar_correction_rhs_d = torch.linalg.solve(train_train_covar_d, test_train_covar_d.transpose(-1, -2))
        return lazify((test_test_covar_d + test_train_covar_d @ covar_correction_rhs_d).float())
