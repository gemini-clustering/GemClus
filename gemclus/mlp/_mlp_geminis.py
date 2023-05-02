from abc import ABC
from numbers import Integral

import numpy as np
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import softmax

from .._base_gemini import _BaseGEMINI, _BaseMMD, _BaseWasserstein


class _MLPGEMINI(_BaseGEMINI, ABC):
    _parameter_constraints: dict = {
        **_BaseGEMINI._parameter_constraints,
        "hidden_dim": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, solver="adam", n_hidden_dim=20, batch_size=None,
                 verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.n_hidden_dim = n_hidden_dim

    def _init_params(self, random_state):
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


class MLPMMD(_MLPGEMINI, _BaseMMD):
    """ Implementation of the maximisation of the MMD-OvA GEMINI using a two-layer neural network as a clustering
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

    kernel: {'additive_chi2', 'chi2', 'cosine','linear','poly','polynomial','rbf','laplacian','sigmoid'},
        default='linear'
        The kernel to use in combination with the MMD objective. It corresponds to one value of `KERNEL_PARAMS`.
        Currently, all kernel parameters are the default ones.

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
        Arnaud Droit, Frederic Preciosio

    See Also
    --------
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
    1.7592155845461592
    """
    _parameter_constraints: dict = {
        **_BaseMMD._parameter_constraints,
        **_MLPGEMINI._parameter_constraints
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, n_hidden_dim=20, kernel="linear", ovo=False,
                 solver="adam", batch_size=None, verbose=False, random_state=None):
        _MLPGEMINI.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_hidden_dim=n_hidden_dim,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        _BaseMMD.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            ovo=ovo,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            kernel=kernel
        )


class MLPWasserstein(_MLPGEMINI, _BaseWasserstein):
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

    metric: {'cosine', 'euclidean', 'l2','l1','manhattan','cityblock'},
        default='euclidean'
        The metric to use in combination with the Wasserstein objective. It corresponds to one value of
        `PAIRED_DISTANCES`.  Currently, all metric parameters are the default ones.

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
        Arnaud Droit, Frederic Preciosio

    See Also
    --------
    MLPMMD: two-layer neural network trained for clustering with the MMD GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.mlp import MLPWasserstein
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = MLPWasserstein(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.7615502432434385
    """
    _parameter_constraints: dict = {
        **_BaseWasserstein._parameter_constraints,
        **_MLPGEMINI._parameter_constraints
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, n_hidden_dim=20, metric="euclidean", ovo=False,
                 solver="adam", batch_size=None, verbose=False, random_state=None):
        _MLPGEMINI.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_hidden_dim=n_hidden_dim,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        _BaseWasserstein.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            ovo=ovo,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            metric=metric
        )
