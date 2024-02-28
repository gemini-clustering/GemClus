import warnings
from abc import ABC
from numbers import Real

import numpy as np
import ot
from sklearn.metrics import pairwise_kernels, pairwise_distances
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, PAIRED_DISTANCES
from sklearn.utils._param_validation import StrOptions, Interval

from .._constraints import constraint_params
from ._base_loss import _GEMINI


class MMDGEMINI(_GEMINI):
    """
    Implements the one-vs-all and one-vs-one MMD GEMINI.
    The one-vs-all version compares the maximum mean discrepancy between a cluster distribution
    and the data distribution.

    The one-vs-one objective is equivalent to a kernel KMeans objective.

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{MMD}_\kappa(p(x|y)\|p(x|y))]

    where :math:`\kappa` is a kernel defined between the samples of the data space.

    The one-vs-one compares the maximum mean discrepancy between two cluster distributions.

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\\text{MMD}_\kappa(p(x|y_a)\|p(x|y_b))]


    Parameters
    ----------
    ovo: bool, default=False
        Whether to use the one-vs-all objective (False) or the one-vs-one objective (True).

    kernel: {'additive_chi2', 'chi2', 'cosine','linear','poly','polynomial','rbf','laplacian','sigmoid', 'precomputed'},
        default='linear'
        The kernel to use in combination with the MMD objective. It corresponds to one value of `KERNEL_PARAMS`.
        Currently, all kernel parameters are the default ones.
        If the kernel is set to 'precomputed', then a custom kernel matrix must be passed to the argument `affinity`
        of the `evaluate` method.

    kernel_params: dict, default=None
        Additional keyword arguments for the kernel function. Ignored if the kernel is callable or precomputed.

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """

    @constraint_params(
        {
            "ovo": [bool],
            "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
            "kernel_params": [dict, None],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, ovo=False, kernel="linear", kernel_params=None, epsilon=1e-12):
        super().__init__(epsilon)
        self.ovo = ovo
        self.kernel_params = kernel_params
        self.kernel = kernel

    def compute_affinity(self, X, y=None):
        """
        Compute the kernel between all samples of X.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The samples between which all affinities must be computed

        y: ndarray of shape (n_samples, n_samples), default=None
            Values of the affinity between samples in case of a "precomputed" affinity. Ignored if None and the affinity
            is not precomputed.

        Returns
        -------
        affinity: ndarray of shape (n_samples, n_samples)
            The kernel between all samples  if it is needed for the GEMINI computations, None otherwise.
        """
        if callable(self.kernel):
            if self.kernel_params is not None:
                warnings.warn("Parameters passed through kernel_params are ignored when kernel is a callable.")
            return self.kernel(X)
        elif self.kernel == "precomputed":
            if y is None:
                raise ValueError(f"Kernel should be precomputed, yet no kernel was passed as parameters: y={y}")
            return y
        _params = dict() if self.kernel_params is None else self.kernel_params
        return pairwise_kernels(X, metric=self.kernel, **_params)

    def evaluate(self, y_pred, affinity, return_grad=False):
        clip_mask = (y_pred > self.epsilon) & (y_pred < (1 - self.epsilon))
        y_pred = np.clip(y_pred, a_min=self.epsilon, a_max=1 - self.epsilon)

        N = y_pred.shape[0]
        normalised_kernel = affinity / N ** 2

        pi = y_pred.mean(0, keepdims=True)

        alpha = y_pred / pi
        gamma = normalised_kernel @ alpha

        if self.ovo:
            omega = alpha.T @ gamma

            A = np.diag(omega).reshape((1, -1))

            delta = np.sqrt(np.maximum(-2 * omega + A + A.T, 0))  # For numerical stability, keep if positive

            mmd_ovo_value = (pi @ delta @ pi.T).squeeze()

            if return_grad:
                # Some distances may be equal to 0 (including self-distances)
                # So we need to remove them from the gradient
                Lambda = (pi.T @ pi) / (delta + np.eye(len(delta)))
                Lambda -= np.diag(np.diag(Lambda))
                Lambda[delta == 0] = 0

                gradient = gamma * Lambda.sum(0, keepdims=True)
                gradient -= gamma @ Lambda
                gradient -= A * Lambda.sum(0, keepdims=True) / N
                gradient += (alpha * (gamma @ Lambda)).mean(0)
                gradient /= pi
                gradient += pi @ delta / N
                gradient *= 2

                return mmd_ovo_value, gradient * clip_mask
            else:
                return mmd_ovo_value
        else:
            omega = alpha * gamma

            a = omega.sum(0)
            b = gamma.sum(0)
            c = normalised_kernel.sum()

            delta = np.sqrt(np.maximum(a + c - 2 * b, 0))  # For numerical stability, keep if positive

            mmd_ova_value = np.dot(pi, delta).squeeze()

            if return_grad:
                tau_grad = (np.eye(N) - 1 / N) @ normalised_kernel @ (alpha - 1)
                delta_mask = (delta == 0)
                gradient = tau_grad / (delta + delta_mask).reshape((1, -1))
                gradient[:, delta_mask] = 0
                return mmd_ova_value, gradient * clip_mask
            else:
                return mmd_ova_value


class WassersteinGEMINI(_GEMINI, ABC):
    """
    Implements the one-vs-all and one-vs-one Wasserstein GEMINI.

    The one-vs-all version compares the Wasserstein distance between a cluster distribution
    and the data distribution.

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\mathcal{W}_\delta(p(x|y)\|p(x|y))]

    where :math:`\delta` is a metric defined between the samples of the data space.

    The one-vs-one version compares the Wasserstein distance between two cluster distributions.

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\mathcal{W}_\delta(p(x|y_a)\|p(x|y_b))]

    Parameters
    ----------
    ovo: bool, default=False
        Whether to use the one-vs-all objective (False) or the one-vs-one objective (True).

    metric: {'cosine', 'euclidean', 'l2','l1','manhattan','cityblock', 'precomputed'}, default='euclidean'
        The metric to use in combination with the Wasserstein objective. It corresponds to one value of
        `PAIRED_DISTANCES`. Currently, all metric parameters are the default ones.
        If the metric is set to 'precomputed', then a custom distance matrix must be passed to the argument `affinity`
        of the `evaluate` method.

    metric_params: dict, default=None
        Additional keyword arguments for the metric function. Ignored if the metric is callable or precomputed.

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """

    @constraint_params(
        {
            "ovo": [bool],
            "metric": [StrOptions(set(list(PAIRED_DISTANCES) + ["precomputed"]))],
            "metric_params": [dict, None],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, ovo=False, metric="euclidean", metric_params=None, epsilon=1e-12):
        super().__init__(epsilon)
        self.ovo = ovo
        self.metric = metric
        self.metric_params = metric_params

    def compute_affinity(self, X, y=None):
        """
        Compute the distance between all samples of X.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The samples between which all affinities must be computed

        y: ndarray of shape (n_samples, n_samples), default=None
            Values of the affinity between samples in case of a "precomputed" affinity. Ignored if None and the affinity
            is not precomputed.

        Returns
        -------
        affinity: ndarray of shape (n_samples, n_samples)
            The distance between all samples if it is needed for the GEMINI computations, None otherwise.
        """
        if callable(self.metric):
            if self.metric_params is not None:
                warnings.warn("Parameters passed through metric_params are ignored when a metric is a callable.")
            return self.metric(X)
        elif self.metric == "precomputed":
            if y is None:
                raise ValueError(f"Kernel should be precomputed, yet no kernel was passed as parameters: y={y}")
            return y
        _params = dict() if self.metric_params is None else self.metric_params
        return pairwise_distances(X, metric=self.metric, **_params)

    def evaluate(self, y_pred, affinity, return_grad=False):
        N, K = y_pred.shape

        clip_mask = (y_pred > self.epsilon) & (y_pred < 1 - self.epsilon)
        y_pred = np.clip(y_pred, a_min=self.epsilon, a_max=1 - self.epsilon)

        pi = y_pred.mean(0)

        wy = np.ascontiguousarray((y_pred / (pi.reshape((1, -1)) * N)).T)

        if self.ovo:
            if return_grad:
                grads = np.zeros(y_pred.shape)
            wasserstein_distances = np.zeros((K, K))

            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    emd, log = ot.emd2(wy[k1], wy[k2], affinity, log=True)
                    wasserstein_distances[k1, k2] = emd
                    wasserstein_distances[k2, k1] = emd

                    if return_grad:
                        u_bar = log["u"] - log["u"].mean()
                        v_bar = log["v"] - log["v"].mean()
                        grads[:, k1] += 2 * pi[k2] * (u_bar / N - (u_bar * y_pred[:, k1] / (N * N * pi[k1])).sum())
                        grads[:, k2] += 2 * pi[k1] * (v_bar / N - (v_bar * y_pred[:, k2] / (N * N * pi[k2])).sum())

            wasserstein_ovo_value = np.dot(pi, np.dot(wasserstein_distances, pi))

            if return_grad:
                grads += 2 * np.dot(wasserstein_distances, pi) / N
                return wasserstein_ovo_value, grads * clip_mask
            else:
                return wasserstein_ovo_value
        else:
            constant_weights = np.ones(N) / N

            wasserstein_distances = np.zeros(y_pred.shape[1])
            dual_variables = [None] * y_pred.shape[1]

            for k in range(K):
                wasserstein_distances[k], dual_variables[k] = ot.emd2(wy[k], constant_weights, affinity, log=True)

            wasserstein_ova_value = np.dot(pi, wasserstein_distances)
            if return_grad:
                u_bar = np.vstack([x["u"] - x["u"].mean() for x in dual_variables]).T
                grads = u_bar / N + wasserstein_distances / N
                grads -= (y_pred * u_bar).sum(0) / (N * N * pi)
                return wasserstein_ova_value, grads * clip_mask
            else:
                return wasserstein_ova_value
