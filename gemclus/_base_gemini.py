"""
This module implements clustering with 2-layer dense neural networks using the MMD OvA/OvO objectives.
"""

from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, PAIRED_DISTANCES, pairwise_kernels, pairwise_distances
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from gemclus._gemini_grads import mmd_ovo, mmd_ova, wasserstein_ovo, wasserstein_ova


class _BaseGEMINI(ClusterMixin, BaseEstimator, ABC):
    """ This is the BaseGEMINI to derive to create a GEMINI MLP or linear clustering model.
     When deriving this class, there are a couple methods to override depending on the design of a discriminative model
     :math:`p(y|x)` or a specific GEMINI.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

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
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    """
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "solver": [StrOptions({"sgd", "adam"})],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "verbose": [bool],
        "random_state": [Interval(Integral, 0, None, closed="left"), None]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, solver="adam", batch_size=None,
                 verbose=False, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.solver = solver
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def _init_params(self, random_state):
        """
        Initialise the set of parameters :math:`\theta` parameters of the model that are used to compute
        :math:`p_\theta(y|x)`.

        Parameters
        ----------
        random_state: RandomState instance
            Determines random number generation for weights and bias initialisation.
        """
        pass

    @abstractmethod
    def _compute_gemini(self, y_pred, K, return_grad=False):
        """
        Compute the GEMINI objective given the predictions :math:`p_\theta(y|x)` and an affinity matrix. The computation
        must return as well the gradients of the GEMINI w.r.t. the predictions. Depending on the context,
        the affinity matrix :ref:`K` can be either a kernel matrix or a distance matrix resulting from the
        :ref:`_compute_affinity` method.

        Parameters
        ----------
        y_pred: ndarray of shape (n_samples, n_clusters)
            The conditional distribution (prediction) of clustering assignment per sample.
        K: ndarray of shape (n_samples, n_samples)
            The affinity matrix resulting from the :ref:`_compute_affinity` method. The matrix must be symmetric.
        return_grad: bool, default=False
            If True, the method should return the gradient of the GEMINI w.r.t. the predictions :math:`p_\theta(y|x)`.

        Returns
        -------
        gemclus: float
            The gemclus score I of the model given the predictions and affinities.
        gradients: ndarray of shape (n_samples, n_clusters)
            The derivative :math:`\nabla_{p_\theta(y|x)}I` of the GEMINI.
        """
        pass

    @abstractmethod
    def _compute_grads(self, X, y_pred, gradient):
        """
        Compute the gradient of each parameter of the model according to the input, output and
        GEMINI gradient of the model. Overall, this method implements model backpropagation.

        Parameters
        -----------
        X: ndarray of shape (n_samples, n_features)
            The passed data inputs from the forward pass
        y_pred: ndarray of shape (n_samples, n_clusters)
            The prediction of the model for the inputs `X`.
        gradient: ndarray of shape (n_samples, n_clusters)
            The gradient of the GEMINI w.r.t. y_pred

        Returns
        --------
        gradients: list of ndarrays of various shapes
            A list containing the gradient for each weight of the model. The gradients must be ordered as the weights
            returned by the method :ref:`_get_weights`.
        """
        pass

    def _update_weights(self, weights, gradients):
        self.optimiser_.update_params(weights, gradients)

    @abstractmethod
    def _get_weights(self):
        """
        Returns all parameters of the model inside a list.

        Returns
        -------
        weights: list of ndarrays of various shapes
            A list containing all parameters of the model.
        """
        pass

    @abstractmethod
    def _infer(self, X, retain=True):
        """
        Perform the forward pass of the model and return the clustering conditional probabilities of the model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The samples to cluster
        retain: bool, default=True
            if True, all intermediate states that will be useful for backpropagation must be saved in attributes.

        Returns
        -------
        y_pred:  ndarray of shape (n_samples, n_clusters)
            The prediction :math:`p_\theta(y|x)` probabilities of the model.
        """
        pass

    @abstractmethod
    def _compute_affinity(self, X, y=None):
        """
        Compute the affinity (kernel function or distance function_ between all samples of X.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The samples between which all affinities must be compute

        y: ndarray of shape (n_samples, n_samples), default=None
            Values of the affinity between samples in case of a "precomputed" affinity. Ignored if None and the affinity
            is not precomputed.

        Return
        ------
        K: ndarray of shape (n_samples, n_samples)
            The symmetric affinity matrix
        """
        pass

    def fit(self, X, y=None):
        """Compute GEMINI clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        y : ndarray of shape (n_samples, n_samples), default=None
            Use this parameter to give a precomputed affinity metric if the option "precomputed" was passed during
            construction. Otherwise, it is not used and present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        # Check that X has the correct shape
        X = check_array(X)
        X = self._validate_data(X, accept_sparse=True, dtype=np.float64, ensure_min_samples=self.n_clusters)

        if self.batch_size is None:
            batch_size = len(X)
        else:
            batch_size = self.batch_size

        # Fix the random seed
        random_state = check_random_state(self.random_state)

        # Initialise the weights
        if self.verbose:
            print("Initialising parameters")
        self._init_params(random_state)
        weights = self._get_weights()

        if self.solver == "sgd":
            self.optimiser_ = SGDOptimizer(weights, self.learning_rate)
        else:
            self.optimiser_ = AdamOptimizer(weights, self.learning_rate)

        if self.verbose:
            print(f"Computing affinity")

        affinity = self._compute_affinity(X, y)

        if self.verbose:
            print(f"Starting training over {self.max_iter} iterations.")
        # Now, iterate for gradient descent
        for i in range(self.max_iter):
            # Create batches
            j = 0
            all_indices = random_state.permutation(len(X))
            while j < len(X):
                batch_indices = all_indices[j:j + batch_size]
                X_batch = X[batch_indices]
                if affinity is None:
                    affinity_batch = None  # Specific case of f-div GEMINI
                else:
                    affinity_batch = affinity[batch_indices][:, batch_indices]

                y_pred = self._infer(X_batch)
                _, grads = self._compute_gemini(y_pred, affinity_batch, return_grad=True)
                grads = self._compute_grads(X_batch, y_pred, grads)
                self._update_weights(weights, grads)

                j += batch_size

        if self.verbose:
            print("Finished")

        # Return the classifier

        # Must save the labels
        self.labels_ = self._infer(X).argmax(1)
        self.n_iter_ = self.max_iter

        return self

    def fit_predict(self, X, y=None):
        """Compute GEMINI clustering and returns the predicted clusters.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        y : ndarray of shape (n_samples, n_samples), default=None
            Use this parameter to give a precomputed affinity metric if the option "precomputed" was passed during
            construction. Otherwise, it is not used and present here for API consistency by convention.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the cluster label for each sample.
        """
        return self.fit(X, y).labels_

    def predict_proba(self, X):
        """
        Probability estimates that are the output of the neural network p(y|x).
        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_clusters)
            Returns the probability of the sample for each cluster in the model.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        y_pred = self._infer(X, retain=False)
        return y_pred

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y=None):

        """
        Return the value of the GEMINI evaluated on the given test data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples, n_samples), default=None
            Use this parameter to give a precomputed affinity metric if the option "precomputed" was passed during
            construction. Otherwise, it is not used and present here for API consistency by convention.

        Returns
        -------
        score : float
            GEMINI evaluated on the output of ``self.predict(X)``.
        """
        K = self._compute_affinity(X, y)
        y_pred = self.predict_proba(X)
        return self._compute_gemini(y_pred, K).item()


class _BaseMMD(_BaseGEMINI, ABC):
    """
    Adds the MMD GEMINI to the base model.

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
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    """
    _parameter_constraints: dict = {
        **_BaseGEMINI._parameter_constraints,
        "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
        "ovo": [bool]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, kernel="linear", batch_size=None,
                 solver="adam", ovo=False, verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.kernel = kernel
        self.ovo = ovo

    def _compute_affinity(self, X, y=None):
        if callable(self.kernel):
            return self.kernel(X)
        elif self.kernel == "precomputed":
            assert y is not None, f"Kernel should be precomputed, yet no kernel was passed as parameters: y={y}"
            return y
        return pairwise_kernels(X, metric=self.kernel)

    def _compute_gemini(self, y_pred, K, return_grad=False):
        if self.ovo:
            return mmd_ovo(y_pred, K, return_grad)
        else:
            return mmd_ova(y_pred, K, return_grad)


class _BaseWasserstein(_BaseGEMINI, ABC):
    """
    Adds the Wasserstein GEMINI to the base model.

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
        **_BaseGEMINI._parameter_constraints,
        "metric": [StrOptions(set(list(PAIRED_DISTANCES) + ["precomputed"]))],
        "ovo": [bool]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, solver="adam", batch_size=None,
                 metric="euclidean", ovo=False, verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.metric = metric
        self.ovo = ovo

    def _compute_affinity(self, X, y=None):
        if callable(self.metric):
            return self.metric(X)
        elif self.metric == "precomputed":
            assert y is not None, f"Kernel should be precomputed, yet no kernel was passed as parameters: y={y}"
            return y
        return pairwise_distances(X, metric=self.metric)

    def _compute_gemini(self, y_pred, K, return_grad=False):
        if self.ovo:
            return wasserstein_ovo(y_pred, K, return_grad)
        else:
            return wasserstein_ova(y_pred, K, return_grad)
