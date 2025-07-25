import warnings
from abc import ABC
from numbers import Real

import numpy as np
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS, pairwise_kernels
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted, check_array, validate_data

from .._base_gemini import DiscriminativeModel
from ..gemini import MMDGEMINI, WassersteinGEMINI


class LinearModel(DiscriminativeModel, ABC):
    """ Implementation of a logistic regression as a clustering distribution :math:`p(y|x)`. Any GEMINI can be
    used to train this model.

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
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    LinearWasserstein: logistic regression trained for clustering with the Wasserstein GEMINI
    LinearMMD: logistic regression trained for clustering with the MMD GEMINI
    RIM: logistic regression trained with a regularised mutual information

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.linear import LinearModel
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = LinearModel(gemini="mmd_ovo", random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.7550724287
    """
    _parameter_constraints: dict = {
        **DiscriminativeModel._parameter_constraints,
    }

    def __init__(self, n_clusters=3, gemini="mmd_ova", max_iter=1000, learning_rate=1e-3, solver="adam",
                 batch_size=None, verbose=False, random_state=None):
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

    def _init_params(self, random_state, X=None):
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
        H = X @ self.W_ + self.b_
        return softmax(H)


class LinearMMD(LinearModel):
    """ Implementation of the maximisation of the MMD GEMINI using a logistic regression as a clustering
    distribution :math:`p(y|x)`.

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
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    LinearModel: logistic regression trained for clustering with any GEMINI
    LinearWasserstein: logistic regression trained for clustering with the Wasserstein GEMINI
    RIM: logistic regression trained with a regularised mutual information

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
    1.7048160115
    """
    _parameter_constraints: dict = {
        **LinearModel._parameter_constraints,
        "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
        "kernel_params": [dict, None],
        "ovo": [bool]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, solver="adam", kernel="linear", ovo=False,
                 batch_size=None, verbose=False, random_state=None, kernel_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=None,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
        )
        self.ovo = ovo
        self.kernel = kernel
        self.kernel_params = kernel_params

    def get_gemini(self):
        return MMDGEMINI(ovo=self.ovo, kernel=self.kernel, kernel_params=self.kernel_params)


class LinearWasserstein(LinearModel):
    """ Implementation of the maximisation of the Wasserstein GEMINI using a logistic regression as a clustering
    distribution :math:`p(y|x)`.

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
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    LinearModel: logistic regression trained for clustering with any GEMINI
    LinearMMD: logistic regression trained for clustering with the MMD GEMINI
    RIM: logistic regression trained with a regularised mutual information

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.linear import LinearWasserstein
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = LinearWasserstein(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.7103992986
    """
    _parameter_constraints: dict = {
        **LinearModel._parameter_constraints,
        "metric": [StrOptions(set(list(PAIRWISE_DISTANCE_FUNCTIONS) + ["precomputed"])), callable],
        "metric_params": [dict, None],
        "ovo": [bool],
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, metric="euclidean", ovo=False,
                 solver="adam", batch_size=None, verbose=False, random_state=None, metric_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=None,
            max_iter=max_iter,
            learning_rate=learning_rate,
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


class RIM(LinearModel):
    r""" Implementation of the maximisation of the classical mutual information using a logistic regression with an
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

        reg: float, default=0.1
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
        LinearModel: logistic regression trained for clustering with any GEMINI
        LinearWasserstein: logistic regression trained for clustering with the Wasserstein GEMINI
        LinearMMD: logistic regression trained for clustering with the MMD GEMINI

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from gemclus.linear import RIM
        >>> X,y=load_iris(return_X_y=True)
        >>> clf = RIM(random_state=0).fit(X)
        >>> clf.predict(X[:2,:])
        array([0, 0])
        >>> clf.predict_proba(X[:2,:]).shape
        (2, 3)
        >>> clf.score(X)
        0.4390485754
        """

    _parameter_constraints: dict = {
        **LinearModel._parameter_constraints,
        "reg": [Interval(Real, 0, None, closed="left")]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, reg=1e-1,
                 solver="adam", batch_size=None, verbose=False, random_state=None):
        LinearModel.__init__(
            self,
            n_clusters=n_clusters,
            gemini="mi",
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.reg = reg

    def _update_weights(self, weights, gradients):
        # Add the regularisation gradient on the weight matrix
        gradients[0] += self.reg * 2 * self.W_
        self.optimiser_.update_params(weights, gradients)


class KernelRIM(LinearModel):
    r""" Implementation of the maximisation of the classical mutual information using a kernelised version of the
        logistic regression with an :math:`\ell_2` penalty on the weights. This implementation follows the framework
        described by Krause et al. in the RIM paper.

        Parameters
        ----------
        n_clusters : int, default=3
            The maximum number of clusters to form as well as the number of output neurons in the neural network.

        max_iter: int, default=1000
            Maximum number of epochs to perform gradient descent in a single run.

        learning_rate: float, default=1e-3
            Initial learning rate used. It controls the step-size in updating the weights.

        reg: float, default=0.1
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

        base_kernel: {'additive_chi2', 'chi2', 'cosine','linear','poly','polynomial','rbf','laplacian','sigmoid'}, or
        callable, default='linear'
            The kernel to use in combination with the MMD objective. It corresponds to one value of `KERNEL_PARAMS`.
            Currently, all kernel parameters are the default ones.

        base_kernel_params: dict, default=None
            A dictionary of keyword arguments to pass to the chosen kernel function.

        Attributes
        ----------
        W_: ndarray of shape (n_samples, n_clusters)
            The linear weights of model for each kernelised sample
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
        LinearModel: logistic regression trained for clustering with any GEMINI
        LinearWasserstein: logistic regression trained for clustering with the Wasserstein GEMINI
        LinearMMD: logistic regression trained for clustering with the MMD GEMINI

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from gemclus.linear import KernelRIM
        >>> X,y=load_iris(return_X_y=True)
        >>> clf = KernelRIM(random_state=0).fit(X)
        >>> clf.predict(X[:2,:])
        array([2, 2])
        >>> clf.predict_proba(X[:2,:]).shape
        (2, 3)
        """
    _parameter_constraints: dict = {
        **DiscriminativeModel._parameter_constraints,
        "reg": [Interval(Real, 0, None, closed="left")],
        "base_kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS))), callable],
        "base_kernel_params": [dict, None]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, reg=1e-1,
                 solver="adam", batch_size=None, verbose=False, random_state=None,
                 base_kernel="linear", base_kernel_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini="mi",
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.reg = reg
        self.base_kernel = base_kernel
        self.base_kernel_params = base_kernel_params

    def _compute_kernel(self, X):
        # Compute the kernel term between X and the input data
        if callable(self.base_kernel):
            if self.base_kernel_params is not None:
                warnings.warn("Parameters passed through kernel_params are ignored when kernel is a callable.")
            kernel = self.base_kernel(X, self.input_data_)
        else:
            _params = dict() if self.base_kernel_params is None else self.base_kernel_params
            kernel = pairwise_kernels(X, self.input_data_, metric=self.base_kernel, **_params)
        return kernel

    def fit(self, X, y=None):
        # We start by storing the input data for later kernel computations
        check_array(X)
        self.input_data_ = X

        training_kernel = self._compute_kernel(X)
        super().fit(training_kernel, y)

        self.n_features_in_ = X.shape[1]

        return self

    def _compute_grads(self, X, y_pred, gradient):
        base_grads = super()._compute_grads(X, y_pred, gradient)
        # Add the regularisation gradient on the weight matrix
        base_grads[0] += 2 * self.reg * np.dot(X, self.W_)
        return base_grads


    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=False, reset=False)
        kernel = self._compute_kernel(X)
        return self._infer(kernel)
