from abc import ABC, abstractmethod
from numbers import Real

import numpy as np
import ot
from sklearn.metrics import pairwise_kernels, pairwise_distances
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, PAIRED_DISTANCES
from sklearn.utils._param_validation import StrOptions, Interval
from .._constraints import constraint_params

AVAILABLE_GEMINIS = ["mmd", "wasserstein", "mi"]


class _GEMINI(ABC):

    def __init__(self, epsilon=1e-12):
        # To stabilise the computations and in order to avoid the
        # softmax overconfident values which would lead to division by 0
        # we, clamp the predictions
        self.epsilon = epsilon

    @abstractmethod
    def evaluate(self, y_pred, affinity, return_grad=False):
        """
        Compute the GEMINI objective given the predictions :math:`$p(y|x)$` and an affinity matrix. The
        computation must return as well the gradients of the GEMINI w.r.t. the predictions. Depending on the context,
        the affinity matrix `affinity` can be either a kernel matrix or a distance matrix resulting from the
        `compute_affinity` method.

        Parameters
        ----------
        y_pred: ndarray of shape (n_samples, n_clusters)
            The conditional distribution (prediction) of clustering assignment per sample.
        affinity: ndarray of shape (n_samples, n_samples)
            The affinity matrix resulting from the `compute_affinity` method. The matrix must be symmetric.
        return_grad: bool, default=False
            If True, the method should return the gradient of the GEMINI w.r.t. the predictions :math:`$p(y|x)$`.

        Returns
        -------
        gemini: float
            The gemini score of the model given the predictions and affinities.

        gradients: ndarray of shape (n_samples, n_clusters)
            The derivative w.r.t. the predictions `y_pred`: :math:`$\\nabla_{p (y|x)} \mathcal{I} $`
        """
        pass

    def __call__(self, y_pred, distance, return_grad=False):
        return self.evaluate(y_pred, distance, return_grad)

    @abstractmethod
    def compute_affinity(self, X, y=None):
        """
        Compute the affinity (kernel function or distance function) between all samples of X. If the GEMINI does not
        compute any affinity, returns None.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The samples between which all affinities must be computed

        y: ndarray of shape (n_samples, n_samples), default=None
            Values of the affinity between samples in case of a "precomputed" affinity. Ignored if None and the affinity
            is not precomputed.

        Returns
        -------
        affinity: ndarray of shape (n_samples, n_samples) or None
            The symmetric affinity matrix if it is needed for the GEMINI computations, None otherwise.
        """
        pass


class MMDGEMINI(_GEMINI, ABC):

    @constraint_params(
        {
            "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, kernel="linear", epsilon=1e-12):
        super().__init__(epsilon)
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
            return self.kernel(X)
        elif self.kernel == "precomputed":
            assert y is not None, f"Kernel should be precomputed, yet no kernel was passed as parameters: y={y}"
            return y
        return pairwise_kernels(X, metric=self.kernel)


class WassersteinGEMINI(_GEMINI, ABC):
    @constraint_params(
        {
            "metric": [StrOptions(set(list(PAIRED_DISTANCES) + ["precomputed"]))],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, metric="euclidean", epsilon=1e-12):
        super().__init__(epsilon)
        self.metric = metric

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
            return self.metric(X)
        elif self.metric == "precomputed":
            assert y is not None, f"Kernel should be precomputed, yet no kernel was passed as parameters: y={y}"
            return y
        return pairwise_distances(X, metric=self.metric)


class MMDOvA(MMDGEMINI):
    """
    Implements the one-vs-all MMD GEMINI which compares the maximum mean discrepancy between a cluster distribution
    and the data distribution.

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{MMD}_\kappa(p(x|y)\|p(x|y))]

    where :math:`\kappa` is a kernel defined between the samples of the data space.

    Parameters
    ----------
    kernel: {'additive_chi2', 'chi2', 'cosine','linear','poly','polynomial','rbf','laplacian','sigmoid', 'precomputed'},
        default='linear'
        The kernel to use in combination with the MMD objective. It corresponds to one value of `KERNEL_PARAMS`.
        Currently, all kernel parameters are the default ones.
        If the kernel is set to 'precomputed', then a custom kernel matrix must be passed to the argument `affinity`
        of the `evaluate` method.

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """
    @constraint_params(
        {
            "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, kernel="linear", epsilon=1e-12):
        super().__init__(kernel, epsilon)

    def evaluate(self, y_pred, affinity, return_grad=False):
        clip_mask = (y_pred > self.epsilon) & (y_pred < (1 - self.epsilon))
        y_pred = np.clip(y_pred, a_min=self.epsilon, a_max=1 - self.epsilon)

        N = y_pred.shape[0]
        normalised_kernel = affinity / N ** 2

        pi = y_pred.mean(0, keepdims=True)

        alpha = y_pred / pi
        gamma = normalised_kernel @ alpha
        omega = alpha * gamma

        a = omega.sum(0)
        b = gamma.sum(0)
        c = normalised_kernel.sum()

        delta = np.sqrt(np.maximum(a + c - 2 * b, 0))  # For numerical stability, keep if positive

        mmd_ova_value = np.dot(pi, delta)

        if return_grad:
            tau_grad = (np.eye(N) - 1 / N) @ normalised_kernel @ (alpha - 1)
            delta_mask = (delta == 0)
            gradient = tau_grad / (delta + delta_mask).reshape((1, -1))
            gradient[:, delta_mask] = 0
            return mmd_ova_value, gradient * clip_mask
        else:
            return mmd_ova_value


class MMDOvO(MMDGEMINI):
    """
    Implements the one-vs-one MMD GEMINI which compares the maximum mean discrepancy between two cluster
    distributions:

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\\text{MMD}_\kappa(p(x|y_a)\|p(x|y_b))]

    where :math:`\kappa` is a kernel defined between the samples of the data space.

    Parameters
    ----------
    kernel: {'additive_chi2', 'chi2', 'cosine','linear','poly','polynomial','rbf','laplacian','sigmoid', 'precomputed'},
        default='linear'
        The kernel to use in combination with the MMD objective. It corresponds to one value of `KERNEL_PARAMS`.
        Currently, all kernel parameters are the default ones.
        If the kernel is set to 'precomputed', then a custom kernel matrix must be passed to the argument `affinity`
        of the `evaluate` method.

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """
    @constraint_params(
        {
            "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, kernel="linear", epsilon=1e-12):
        super().__init__(kernel, epsilon)

    def evaluate(self, y_pred, affinity, return_grad=False):
        clip_mask = (y_pred > self.epsilon) & (y_pred < 1 - self.epsilon)
        y_pred = np.clip(y_pred, a_min=self.epsilon, a_max=1 - self.epsilon)

        N = y_pred.shape[0]
        normalised_kernel = affinity / N ** 2

        pi = y_pred.mean(0, keepdims=True)

        alpha = y_pred / pi
        gamma = normalised_kernel @ alpha
        omega = alpha.T @ gamma

        A = np.diag(omega).reshape((1, -1))

        delta = np.sqrt(np.maximum(-2 * omega + A + A.T, 0))  # For numerical stability, keep if positive

        mmd_ovo_value = pi @ delta @ pi.T

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


class WassersteinOvA(WassersteinGEMINI):
    """
    Implements the one-vs-all Wasserstein GEMINI which compares the Wasserstein distance between a cluster distribution
    and the data distribution.

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\mathcal{W}_\delta(p(x|y)\|p(x|y))]

    where :math:`\delta` is a metric defined between the samples of the data space.

    Parameters
    ----------
    metric: {'cosine', 'euclidean', 'l2','l1','manhattan','cityblock', 'precomputed'}, default='euclidean'
        The metric to use in combination with the Wasserstein objective. It corresponds to one value of
        `PAIRED_DISTANCES`. Currently, all metric parameters are the default ones.
        If the metric is set to 'precomputed', then a custom distance matrix must be passed to the argument `affinity`
        of the `evaluate` method.

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """
    @constraint_params(
        {
            "metric": [StrOptions(set(list(PAIRED_DISTANCES) + ["precomputed"]))],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, metric="euclidean", epsilon=1e-12):
        super().__init__(metric, epsilon)

    def evaluate(self, y_pred, affinity, return_grad=False):
        N, K = y_pred.shape

        clip_mask = (y_pred > self.epsilon) & (y_pred < 1 - self.epsilon)
        y_pred = np.clip(y_pred, a_min=self.epsilon, a_max=1 - self.epsilon)

        pi = y_pred.mean(0)

        wy = np.ascontiguousarray((y_pred / (pi.reshape((1, -1)) * N)).T)

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


class WassersteinOvO(WassersteinGEMINI):
    """
    Implements the one-vs-one Wasserstein GEMINI which compares the Wasserstein distance between two cluster
    distributions:

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\mathcal{W}_\delta(p(x|y_a)\|p(x|y_b))]

    where :math:`\delta` is a metric defined between the samples of the data space.

    Parameters
    ----------
    metric: {'cosine', 'euclidean', 'l2','l1','manhattan','cityblock', 'precomputed'}, default='euclidean'
        The metric to use in combination with the Wasserstein objective. It corresponds to one value of
        `PAIRED_DISTANCES`. Currently, all metric parameters are the default ones.
        If the metric is set to 'precomputed', then a custom distance matrix must be passed to the argument `affinity`
        of the `evaluate` method.

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """
    @constraint_params(
        {
            "metric": [StrOptions(set(list(PAIRED_DISTANCES) + ["precomputed"]))],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, metric="euclidean", epsilon=1e-12):
        super().__init__(metric, epsilon)

    def evaluate(self, y_pred, affinity, return_grad=False):
        N, K = y_pred.shape

        if return_grad:
            grads = np.zeros(y_pred.shape)

        clip_mask = (y_pred > self.epsilon) & (y_pred < 1 - self.epsilon)
        y_pred = np.clip(y_pred, a_min=self.epsilon, a_max=1 - self.epsilon)

        pi = y_pred.mean(0)

        wy = np.ascontiguousarray((y_pred / (pi.reshape((1, -1)) * N)).T)

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


class MI(_GEMINI):
    """
    Implements the classical mutual information between cluster conditional probabilities and the complete data
    probabilities:

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{KL}(p(x|y)\|p(x))]

    Parameters
    ----------
    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """

    def __init__(self, epsilon=1e-12):
        super().__init__(epsilon)

    def evaluate(self, y_pred, affinity, return_grad=False):
        # Start by computing mutual information
        p_y_x = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        p_y = p_y_x.mean(0)

        log_p_y_x = np.log(p_y_x)
        log_p_y = np.log(p_y)

        cluster_entropy = np.sum(p_y * log_p_y)
        prediction_entropy = np.sum(np.mean(p_y_x * log_p_y_x, axis=0))

        mutual_information = prediction_entropy - cluster_entropy

        if return_grad:
            gradient_mi = -log_p_y_x / log_p_y_x.shape[0] + log_p_y
            return mutual_information, -gradient_mi
        else:
            return mutual_information

    def compute_affinity(self, X, y=None):
        return None
