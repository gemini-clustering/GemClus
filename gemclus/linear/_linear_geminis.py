from abc import ABC
from numbers import Real

import numpy as np
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import softmax

from .._base_gemini import _BaseGEMINI, _BaseMMD, _BaseWasserstein


class _LinearGEMINI(_BaseGEMINI, ABC):
    _parameter_constraints: dict = {
        **_BaseGEMINI._parameter_constraints,
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, solver="adam", batch_size=None,
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

    def _init_params(self, random_state):
        in_threshold = np.sqrt(1 / self.n_features_in_)
        self.W_ = random_state.uniform(-in_threshold, in_threshold, size=(self.n_features_in_, self.n_clusters))
        self.b_ = random_state.uniform(-in_threshold, in_threshold, size=(1, self.n_clusters))

    def _compute_grads(self, X, y_pred, gradient):
        tau_hat_grad = y_pred * (gradient - (y_pred * gradient).sum(1, keepdims=True))  # Shape NxK

        W_grad = X.T @ tau_hat_grad
        b_grad = tau_hat_grad.sum(0, keepdims=True)

        # Negative sign to force the optimiser to maximise instead of minimise
        gradients = [-W_grad, -b_grad]

        return gradients

    def _get_weights(self):
        return [self.W_, self.b_]

    def _infer(self, X, retain=True):
        H = np.maximum((X @ self.W_ + self.b_), 0)
        return softmax(H)


class LinearMMD(_LinearGEMINI, _BaseMMD):
    """ Implementation of the maximisation of the MMD-OvA GEMINI using a logistic regression as a clustering
    distribution :math:`p(y|x)`.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

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
    W_: ndarray of shape (n_features, n_clusters)
        The linear weights of model
    b_: ndarray of shape (1, n_clusters)
        The biases of the model
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Preciosio

    See Also
    --------
    LinearWasserstein: logistic regression trained for clustering with the Wasserstein GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.linear import LinearMMD
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = LinearMMD(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.6949190522657158
    """
    _parameter_constraints: dict = {
        **_BaseMMD._parameter_constraints,
        **_LinearGEMINI._parameter_constraints
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, kernel="linear", solver="adam", ovo=False,
                 batch_size=None, verbose=False, random_state=None):
        _BaseMMD.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            ovo=ovo,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            kernel=kernel
        )


class LinearWasserstein(_LinearGEMINI, _BaseWasserstein):
    """ Implementation of the maximisation of the Wasserstein GEMINI using a logisti regression as a clustering
    distribution :math:`p(y|x)`.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    metric: {'cosine', 'euclidean', 'l2','l1','manhattan','cityblock'},
        default='euclidean'
        The metric to use in combination with the Wasserstein objective. It corresponds to one value of
        `PAIRED_DISTANCES`. Currently, all metric parameters are the default ones.

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
    W_: ndarray of shape (n_features_in, n_clusters)
        The linear weights of model
    b_: ndarray of shape (1, n_clusters)
        The biases of the model
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Preciosio

    See Also
    --------
    LinearMMD: logistic regression trained for clustering with the MMD GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.linear import LinearWasserstein
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = LinearWasserstein(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([2, 2])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.6509087196143133
    """
    _parameter_constraints: dict = {
        **_BaseWasserstein._parameter_constraints,
        **_LinearGEMINI._parameter_constraints
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, metric="euclidean", ovo=False,
                 solver="adam", batch_size=None, verbose=False, random_state=None):
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


class RIM(_LinearGEMINI):
    """ Implementation of the maximisation of the classical mutual information using a logistic regression with an
    :math:`\ell_2` penalty on the weights. This implementation follows the framework described by Krause et al. in the
    RIM paper.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    reg: float, default=1.0
        Regularisation hyperparameter for the $\ell_2$ weight penalty.

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
    W_: ndarray of shape (n_features_in, n_clusters)
        The linear weights of model
    b_: ndarray of shape (1, n_clusters)
        The biases of the model
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.

    References
    ----------
    RIM - Discriminative Clustering by Regularized Information Maximization
        Ryan Gomes, Andreas Krause, Pietro Perona. 2010.

    See Also
    --------
    LinearMMD: logistic regression trained for clustering with the MMD GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.linear import RIM
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = RIM(learning_rate=1e-2, random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    0.00962912118121384
    """

    _parameter_constraints: dict = {
        **_LinearGEMINI._parameter_constraints,
        "reg": [Interval(Real, 0, None, closed="left")]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, reg=1,
                 solver="adam", batch_size=None, verbose=False, random_state=None):
        _LinearGEMINI.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.reg = reg

    def _compute_affinity(self, X, y=None):
        return None

    def _compute_gemini(self, y_pred, K, return_grad=False):
        # Start by computing mutual information
        p_y_x = np.clip(y_pred, 1e-12, 1 - 1e-12)
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

    def compute_penalty(self):
        return self.reg * np.sum(self.W_ * self.W_)

    def _update_weights(self, weights, gradients):
        # Add the regularisation gradient on the weight matrix
        gradients[0] += self.reg * 2 * self.W_
        self.optimiser_.update_params(weights, gradients)
