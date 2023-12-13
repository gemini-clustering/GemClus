from abc import ABC, abstractmethod


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

