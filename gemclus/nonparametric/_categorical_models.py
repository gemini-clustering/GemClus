from abc import ABC

from sklearn.utils.extmath import softmax

from gemclus._base_gemini import _BaseGEMINI, _BaseMMD, _BaseWasserstein


class _CategoricalGEMINI(_BaseGEMINI, ABC):
    _parameter_constraints: dict = {
        **_BaseGEMINI._parameter_constraints,
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, solver="adam",
                 verbose=False, random_state=None):
        _BaseGEMINI.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            verbose=verbose,
            random_state=random_state
        )

    def _init_params(self, random_state, X=None):
        self.logits_ = random_state.uniform(-1, 1, size=(len(X), self.n_clusters))

    def _compute_grads(self, X, y_pred, gradient):
        tau_hat_grad = y_pred * (gradient - (y_pred * gradient).sum(1, keepdims=True))  # Shape NxK

        return [-tau_hat_grad]

    def _get_weights(self):
        return [self.logits_]

    def _infer(self, X, retain=True):
        return softmax(self.logits_)



    def _batchify(self, X, affinity_matrix=None, random_state=None):
        """
        Yields elements of X and its corresponding affinity matrix in batches with a uniform random sampling.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training instances to cluster

        affinity_matrix: ndarray of shape (n_samples, n_samples) or None
            The affinity computed between all elements of `X`. Setting to None means the GEMINI doe snot need any
            affinity.

        random_state: int, RandomState instance, default=None
            Unused, here for legacy.

        Returns
        -------
        X_batch: ndarray of shape (n_batch, n_features)
            The batch of data elements

        affinity_batch: ndarray of shape (n_batch, n_batch) or None
            The affinity values of the corresponding elements of the data batch. If the parameter `affinity_matrix` was
            None, then None is returned.
        """
        yield X, affinity_matrix


class CategoricalMMD(_CategoricalGEMINI, _BaseMMD):
    """ The CategoricalMMD is a nonparametric model where each sample is directly assign a probability vector of
    as conditional clustering distribution. Consequently, the parameters do not depend on the value of :math:`$x$`.

    .. math::
        p(y=k|x_i) = \\theta_{ki}

    Contrarily to other models, the categorical distribution can not be used for clustering samples that were not
    part of the training set and does not support batching as well.

    The model optimises the parameters to maximise the MMD GEMINI.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    kernel: {'additive_chi2', 'chi2', 'cosine','linear','poly','polynomial','rbf','laplacian','sigmoid', 'precomputed'},
        default='linear'
        The kernel to use in combination with the MMD objective. It corresponds to one value of `KERNEL_PARAMS`.
        Currently, all kernel parameters are the default ones.
        If the kernel is set to 'precomputed', then a custom kernel matrix must be passed to the argument `y` of
        `fit`, `fit_predict` and/or `score`.

    ovo: bool, default=False
        Whether to run the model using the MMD OvA (False) or the MMD OvO (True).

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for weights and bias initialisation.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    """
    _parameter_constraints: dict = {
        **_CategoricalGEMINI._parameter_constraints,
        **_BaseMMD._parameter_constraints,
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, kernel="linear", solver="adam", ovo=False,
                 verbose=False, random_state=None):
        _BaseMMD.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            ovo=ovo,
            solver=solver,
            verbose=verbose,
            random_state=random_state,
            kernel=kernel
        )


class CategoricalWasserstein(_CategoricalGEMINI, _BaseWasserstein):
    """ The CategoricalWasserstein is a nonparametric model where each sample is directly assign a probability vector of
    as conditional clustering distribution. Consequently, the parameters do not depend on the value of :math:`$x$`.

    .. math::
        p_(y=k|x_i) = \\theta_{ki}

    Contrarily to other models, the categorical distribution can not be used for clustering samples that were not
    part of the training set and does not support batching as well.

    The model optimises the parameters to maximise the Wasserstein GEMINI.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    metric: {'cosine', 'euclidean', 'l2','l1','manhattan','cityblock', 'precomputed'},
        default='euclidean'
        The metric to use in combination with the Wasserstein objective. It corresponds to one value of
        `PAIRED_DISTANCES`. Currently, all metric parameters are the default ones.
        If the metric is set to 'precomputed', then a custom distance matrix must be passed to the argument `y` of
        `fit`, `fit_predict` and/or `score`.

    ovo: bool, default=False
        Whether to run the model using the Wasserstein OvA (False) or the Wasserstein OvO (True).

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for weights and bias initialisation.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    """
    _parameter_constraints: dict = {
        **_CategoricalGEMINI._parameter_constraints,
        **_BaseWasserstein._parameter_constraints,
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, metric="euclidean", ovo=False,
                 solver="adam", verbose=False, random_state=None):
        _BaseWasserstein.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            ovo=ovo,
            verbose=verbose,
            random_state=random_state,
            metric=metric
        )
