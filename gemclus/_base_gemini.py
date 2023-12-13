"""
This module implements clustering with 2-layer dense neural networks using the MMD OvA/OvO objectives.
"""

from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from gemclus.gemini import AVAILABLE_GEMINIS
from .gemini._base_loss import _GEMINI
from .gemini._utils import _str_to_gemini


class DiscriminativeModel(ClusterMixin, BaseEstimator, ABC):
    """ This is the BaseGEMINI to derive to create a GEMINI MLP or linear clustering model.
     When deriving this class, there are a couple methods to override depending on the design of a discriminative model
     :math:`p(y|x)` or a specific GEMINI.

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
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    """
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "gemini": [StrOptions(set(AVAILABLE_GEMINIS)), _GEMINI, None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "solver": [StrOptions({"sgd", "adam"})],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "verbose": [bool],
        "random_state": [Interval(Integral, 0, None, closed="left"), None]
    }

    def __init__(self, n_clusters=3, gemini="mmd_ova", max_iter=1000, learning_rate=1e-3, solver="adam",
                 batch_size=None, verbose=False, random_state=None):
        self.n_clusters = n_clusters
        self.gemini = gemini
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.solver = solver
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def _init_params(self, random_state, X=None):
        """
        Initialise the set of parameters :math:`\theta` parameters of the model that are used to compute
        :math:`p_\theta(y|x)`.

        Parameters
        ----------
        random_state: RandomState instance
            Determines random number generation for weights and bias initialisation.

        X: ndarray of shape (n_samples, n_features), default=None
            The data to fit in case it is needed for a special initialisation of the weights.
        """
        pass

    def get_gemini(self):
        """
        Initialise a :class:`gemclus.GEMINI` instance that will be used to train the model.

        Returns
        -------
        gemini: a GEMINI instance
        """
        if self.gemini is None:
            return _str_to_gemini("mmd_ova")
        if isinstance(self.gemini, str):
            return _str_to_gemini(self.gemini)
        return self.gemini

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

    def _batchify(self, X, affinity_matrix=None, random_state=None):
        """
        Yield elements of X and its corresponding affinity matrix in batches with a uniform random sampling.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training instances to cluster

        affinity_matrix: ndarray of shape (n_samples, n_samples) or None
            The affinity computed between all elements of `X`. Setting to None means the GEMINI doe snot need any
            affinity.

        random_state: int, RandomState instance, default=None

        Returns
        -------
        X_batch: ndarray of shape (n_batch, n_features)
            The batch of data elements

        affinity_batch: ndarray of shape (n_batch, n_batch) or None
            The affinity values of the corresponding elements of the data batch. If the parameter `affinity_matrix` was
            None, then None is returned.
        """
        random_state = check_random_state(random_state)
        all_indices = random_state.permutation(len(X))
        batch_size = len(X) if self.batch_size is None else self.batch_size
        j = 0
        while j < len(X):
            batch_indices = all_indices[j:j + batch_size]
            X_batch = X[batch_indices]
            if affinity_matrix is not None:
                affinity_batch = affinity_matrix[batch_indices][:, batch_indices]
            else:
                affinity_batch = None
            yield X_batch, affinity_batch
            j += batch_size

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

        # Fix the random seed
        random_state = check_random_state(self.random_state)

        # Initialise the weights
        if self.verbose:
            print("Initialising parameters")
        self._init_params(random_state, X)
        weights = self._get_weights()
        gemini = self.get_gemini()

        if self.solver == "sgd":
            self.optimiser_ = SGDOptimizer(weights, self.learning_rate)
        else:
            self.optimiser_ = AdamOptimizer(weights, self.learning_rate)

        if self.verbose:
            print(f"Computing affinity")

        affinity = gemini.compute_affinity(X, y)

        if self.verbose:
            print(f"Starting training over {self.max_iter} iterations.")
        # Now, iterate for gradient descent
        for i in range(self.max_iter):
            # Create batches
            for X_batch, affinity_batch in self._batchify(X, affinity, random_state):
                y_pred = self._infer(X_batch)
                _, grads = gemini(y_pred, affinity_batch, return_grad=True)
                grads = self._compute_grads(X_batch, y_pred, grads)
                self._update_weights(weights, grads)

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
        """ Return the cluster membership of samples. This can only be called after the model was fit to some data.

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
        gemini = self.get_gemini()
        K = gemini.compute_affinity(X, y)
        y_pred = self.predict_proba(X)
        return gemini(y_pred, K).item()

