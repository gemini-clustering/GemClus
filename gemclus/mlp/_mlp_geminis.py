from numbers import Integral

import numpy as np
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import softmax

from .._base_gemini import DiscriminativeModel
from ..gemini import MMDGEMINI, WassersteinGEMINI


class MLPModel(DiscriminativeModel):
    """ Implementation of a two-layer neural network as a clustering distribution :math:`p(y|x)`. Any GEMINI can be used
    to train this model.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    gemini: str, GEMINI instance or None, default="mmd_ova"
        GEMINI objective used to train this discriminative model. Can be "mmd_ova", "mmd_ovo", "wasserstein_ova",
        "wasserstein_ovo", "mi" or other GEMINI available in `gemclus.gemini.AVAILABLE_GEMINI`. Default GEMINIs
        involve the Euclidean metric or linear kernel. To incorporate custom metrics, a GEMINI can also
        be passed as an instance. If set to None, the GEMINI will be MMD OvA with linear kernel.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    n_hidden_dim: int, default=20
        The number of neurons in the hidden layer of the neural network.

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    batch_size: int, default=None
        The size of batches during gradient descent training. If set to None, the whole data will be considered.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for weights and bias initialisation.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    W1_: ndarray, shape (n_features, n_hidden_dim)
        The linear weights of the first layer
    b1_: ndarray of shape (1, n_hidden_dim)
        The biases of the first layer
    W2_: ndarray of shape (n_hidden_dim, n_clusters)
        The linear weights of the hidden layer
    b2_: ndarray of shape (1, n_clusters)
        The biases of the hidden layer
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    H_: ndarray of shape (n_samples, n_hidden_dim)
        The hidden representation of the samples after fitting.OvA

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    MLPMMD: two-layer neural network trained for clustering with the MMD GEMINI
    MLPWasserstein: two-layer neural network trained for clustering with the Wasserstein GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.mlp import MLPModel
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = MLPModel(gemini="mi",random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    0.6325592616
    """
    _parameter_constraints: dict = {
        **DiscriminativeModel._parameter_constraints,
        "n_hidden_dim": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, n_clusters=3, gemini="mmd_ova", max_iter=1000, learning_rate=1e-3, solver="adam",
                 n_hidden_dim=20, batch_size=None, verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=gemini,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.n_hidden_dim = n_hidden_dim

    def _init_params(self, random_state, X=None):
        in_threshold = np.sqrt(1 / self.n_features_in_)
        hidden_threshold = np.sqrt(1 / self.n_hidden_dim)
        self.W1_ = random_state.uniform(-in_threshold, in_threshold, size=(self.n_features_in_, self.n_hidden_dim))
        self.b1_ = random_state.uniform(-in_threshold, in_threshold, size=(1, self.n_hidden_dim))
        self.W2_ = random_state.uniform(-hidden_threshold, hidden_threshold, size=(self.n_hidden_dim, self.n_clusters))
        self.b2_ = random_state.uniform(-hidden_threshold, hidden_threshold, size=(1, self.n_clusters))

    def _compute_grads(self, X, y_pred, gradient):
        tau_hat_grad = y_pred * (gradient - (y_pred * gradient).sum(1, keepdims=True))  # Shape NxK

        W2_grad = self.H_.T @ tau_hat_grad  # Shape
        b2_grad = tau_hat_grad.sum(0, keepdims=True)

        backprop_grad = tau_hat_grad @ W2_grad.T
        backprop_grad *= self.H_ > 0
        W1_grad = X.T @ backprop_grad
        b1_grad = backprop_grad.sum(0, keepdims=True)

        # Negative sign to force the optimiser to maximise instead of minimise
        gradients = [-W1_grad, -W2_grad, -b1_grad, -b2_grad]

        return gradients

    def _get_weights(self):
        return [self.W1_, self.W2_, self.b1_, self.b2_]

    def _infer(self, X, retain=True):
        H = np.maximum((X @ self.W1_ + self.b1_), 0)
        if retain:
            self.H_ = H
        return softmax(H @ self.W2_ + self.b2_)


class MLPMMD(MLPModel):
    """ Implementation of the maximisation of the MMD GEMINI using a two-layer neural network as a clustering
    distribution :math:`p(y|x)`.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    n_hidden_dim: int, default=20
        The number of neurons in the hidden layer of the neural network.

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

    batch_size: int, default=None
        The size of batches during gradient descent training. If set to None, the whole data will be considered.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for weights and bias initialisation.
        Pass an int for reproducible results across multiple function calls.

    kernel_params: dict, default=None
        A dictionary of keyword arguments to pass to the chosen kernel function.

    Attributes
    ----------
    W1_: ndarray, shape (n_features, n_hidden_dim)
        The linear weights of the first layer
    b1_: ndarray of shape (1, n_hidden_dim)
        The biases of the first layer
    W2_: ndarray of shape (n_hidden_dim, n_clusters)
        The linear weights of the hidden layer
    b2_: ndarray of shape (1, n_clusters)
        The biases of the hidden layer
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    H_: ndarray of shape (n_samples, n_hidden_dim)
        The hidden representation of the samples after fitting.OvA

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    MLPModel: two-layer neural network trained for clustering with any GEMINI
    MLPWasserstein: two-layer neural network trained for clustering with the Wasserstein GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.mlp import MLPMMD
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = MLPMMD(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([2, 2])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.7592155845
    """
    _parameter_constraints: dict = {
        **MLPModel._parameter_constraints,
        "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
        "kernel_params": [dict, None],
        "ovo": [bool]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, n_hidden_dim=20, kernel="linear", ovo=False,
                 solver="adam", batch_size=None, verbose=False, random_state=None, kernel_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=None,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_hidden_dim=n_hidden_dim,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.ovo = ovo
        self.kernel = kernel
        self.kernel_params = kernel_params

    def get_gemini(self):
        return MMDGEMINI(ovo=self.ovo, kernel=self.kernel, kernel_params=self.kernel_params)


class MLPWasserstein(MLPModel):
    """ Implementation of the maximisation of the Wasserstein GEMINI using a two-layer neural network as a clustering
    distribution :math:`p(y|x)`.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    n_hidden_dim: int, default=20
        The number of neurons in the hidden layer of the neural network.

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

    batch_size: int, default=None
        The size of batches during gradient descent training. If set to None, the whole data will be considered.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for weights and bias initialisation.
        Pass an int for reproducible results across multiple function calls.

    metric_params: dict, default=None
        A dictionary of keyword arguments to pass to the chosen metric function.

    Attributes
    ----------
    W1_: ndarray, shape (n_features, n_hidden_dim)
        The linear weights of the first layer
    b1_: ndarray of shape (1, n_hidden_dim)
        The biases of the first layer
    W2_: ndarray of shape (n_hidden_dim, n_clusters)
        The linear weights of the hidden layer
    b2_: ndarray of shape (1, n_clusters)
        The biases of the hidden layer
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    H_: ndarray of shape (n_samples, n_hidden_dim)
        The hidden representation of the samples after fitting.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    MLPModel: two-layer neural network trained for clustering with any GEMINI
    MLPMMD: two-layer neural network trained for clustering with the MMD GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.mlp import MLPWasserstein
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = MLPWasserstein(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([2, 2])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.7636482655
    """
    _parameter_constraints: dict = {
        **MLPModel._parameter_constraints,
        "metric": [StrOptions(set(list(PAIRWISE_DISTANCE_FUNCTIONS) + ["precomputed"])), callable],
        "metric_params": [dict, None],
        "ovo": [bool],
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, n_hidden_dim=20, metric="euclidean", ovo=False,
                 solver="adam", batch_size=None, verbose=False, random_state=None, metric_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=None,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_hidden_dim=n_hidden_dim,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.ovo = ovo
        self.metric = metric
        self.metric_params = metric_params

    def get_gemini(self):
        return WassersteinGEMINI(ovo=self.ovo, metric=self.metric, metric_params=self.metric_params)
