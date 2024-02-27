import warnings
from numbers import Real

import numpy as np
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer
from sklearn.utils._param_validation import Interval, StrOptions

from ._base_sparse import _path, check_groups
from ._prox_grad import linear_prox_grad, group_linear_prox_grad
from ..gemini import MMDGEMINI
from ..linear._linear_geminis import LinearModel


class SparseLinearModel(LinearModel):
    """ This is the SparseLinearModel clustering model.
    When deriving, the only methods to adapt is the _compute_gemini methods which
    should be able to return the gradient with respect to the conditional distribution p(y|x).

    On top of the vanilla Linear GEMINI model, this variation brings a group-lasso penalty constraint to ensure
    feature selection via a proximal gradient during training.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    gemini: str, GEMINI instance or None, default="mmd_ova"
        GEMINI objective used to train this discriminative model. Can be "mmd_ova", "mmd_ovo", "wasserstein_ova",
        "wasserstein_ovo", "mi" or other GEMINI available in `gemclus.gemini.AVAILABLE_GEMINI`. Default GEMINIs
        involve the Euclidean metric or linear kernel. To incorporate custom metrics, a GEMINI can also
        be passed as an instance. If set to None, the GEMINI will be MMD OvA with linear kernel.

    groups: list of arrays of various shapes, default=None
        If groups is set, it must describe a partition of the indices of variables. This will be used for performing
        variable selection with groups of features considered to represent one variable. This option can typically be
        used for one-hot-encoded variables. Variable indices that are not entered will be considered alone.
        For example, with 3 features, accepted values can be [[0],[1],[2]], [[0,1],[2]] or [[0,1]].

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    dynamic: bool, default=False
        Whether to run the path in dynamic mode or not. The dynamic mode consists of affinities computed using
        only the subset of selected variables instead of all variables.

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    alpha: float, default=1e-2
        The weight of the group-lasso penalty in the optimisation scheme.

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
    groups_: list of lists of int or None
        The explicit partition of the variables formed by the groups parameter if it was not None.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso
    Sparse GEMINI - Sparse GEMINI for joint discriminative clustering and feature selection
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso


    See Also
    --------
    SparseMLPModel: sparse two-layer neural network trained with any GEMINI
    SparseMLPMMD: sparse two-layer neural network trained for clustering with the MMD GEMINI

    """
    _parameter_constraints: dict = {
        **LinearModel._parameter_constraints,
        "alpha": [Interval(Real, 0, np.inf, closed="left")],
        "groups": [list, None],
        "dynamic": [bool]
    }

    def __init__(self, n_clusters=3, gemini="mmd_ova", groups=None, max_iter=1000, learning_rate=1e-3, alpha=1e-2,
                 batch_size=None, dynamic=False, solver="adam", verbose=False, random_state=None):
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
        self.alpha = alpha
        self.groups = groups
        self.dynamic = dynamic

    def _update_weights(self, weights, gradients):
        # First update the weights according to our optimiser
        self.optimiser_.update_params(weights, gradients)

        # Then statisfy the sparsity constraint of the MLP by
        # evaluating the proximal gradient
        if self.groups_ is None:
            new_W = linear_prox_grad(self.W_, self.alpha * self.optimiser_.learning_rate)
        else:
            new_W = group_linear_prox_grad(self.groups_, self.W_, self.alpha * self.optimiser_.learning_rate)

        np.copyto(self.W_, new_W)

    def _n_selected_features(self):
        return (np.linalg.norm(self.W_, axis=1, ord=2) != 0).sum()

    def get_selection(self):
        """
        Retrieves the indices of features that were selected by the model.

        Returns
        -------
        ind: ndarray
            The indices of the selected features.
        """
        return np.nonzero(np.linalg.norm(self.W_, axis=1, ord=2))[0]

    def _group_lasso_penalty(self):
        return np.linalg.norm(self.W_, axis=1, ord=2).sum()

    def fit(self, X, y=None):
        self._validate_data(X)
        self.groups_ = check_groups(self.groups, X.shape[1])  # Intercept to check that group forms a partition
        return super().fit(X, y)

    def path(self, X, y=None, alpha_multiplier=1.05, min_features=2, keep_threshold=0.9, restore_best_weights=True,
             early_stopping_factor=0.99, max_patience=10):
        """
        Unfold the progressive geometric increase of the penalty weight starting from the initial alpha until
        there remains only a specified amount of features.

        The history of the different gemini scores are kept as well as the best weights with minimum of features
        ensuring that the GEMINI score remains at a certain percentage of the maximum GEMINI score seen during the
        path.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples on which the feature reduction will be made.
        y : ndarray of shape (n_samples, n_samples), default=None
            Use this parameter to give a precomputed affinity metric if the option "precomputed" was passed during
            construction. Otherwise, it is not used. This parameter is incompatible with the dynamic mode.
        alpha_multiplier : float, default=1.05
            The geometric increase of the group-lasso penalty at each-retraining. It must be greater than 1.
        min_features: int, default=2
            The number of features that must remain at best to stop performing the path.
        keep_threshold: float, default=0.9
            The percentage of the maximal GEMINI under which any solution with a minimal number of features is deemed
            best.
        restore_best_weights: bool, default=True
            After performing the path, the best weights offering simultaneously good GEMINI score and few features
            are restored to the model. If the model is set to `dynamic=True`, then this option will be ignored because
            of the incomparable nature of GEMINIs when the number of selected variables change.
        early_stopping_factor: float, default=0.99
            The percentage factor beyond which upgrades of the GEMINI or the group-lasso penalty are considered
            too small for early stopping.
        max_patience:
            The maximum number of iterations to wait without any improvements in either the gemini score or the
            group-lasso penalty before stopping the current step.

        Returns
        -------
        best_weights: list of ndarray of various shapes of length 5
            The list containing the best weights during the path. Sequentially: `W_`, `b_`
        geminis: list of float of length T
            The history of the gemini scores as the penalty alpha was increased.
        group_penalties: list of float of length T
            The history of the group-lasso penalties
        alphas: list of float of length T
            The history of the penalty alphas during the path.
        n_features: list of float of length T
            The number of features that were selected at step t.
        """
        if y is not None and self.dynamic:
            warnings.warn("Dynamic mode is incompatible with a precomputed metric. Ignoring dynamic mode.")

        best_weights, geminis, group_lasso_penalties, alphas, n_features = _path(self, X, y, alpha_multiplier,
                                                                                 min_features, keep_threshold,
                                                                                 early_stopping_factor, max_patience)

        if restore_best_weights:
            if not self.dynamic:
                if self.verbose:
                    print("Restoring best weights")
                np.copyto(self.W_, best_weights[0])
                np.copyto(self.b_, best_weights[1])
            else:
                warnings.warn("The option restore_best_weights is incompatible with the dynamic mode. The final model "
                              "of the path will be kept.")

        return best_weights, geminis, group_lasso_penalties, alphas, n_features


class SparseLinearMMD(SparseLinearModel):
    """ Trains a logistic regression with sparse parameters using the MMD GEMINI.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    groups: list of arrays of various shapes, default=None
        If groups is set, it must describe a partition of the indices of variables. This will be used for performing
        variable selection with groups of features considered to represent one variable. This option can typically be
        used for one-hot-encoded variables. Variable indices that are not entered will be considered alone.
        For example, with 3 features, accepted values can be [[0],[1],[2]], [[0,1],[2]] or [[0,1]].
        
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

    dynamic: bool, default=False
        Whether to run the path in dynamic mode or not. The dynamic mode consists of affinities computed using
        only the subset of selected variables instead of all variables.

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    alpha: float, default=1e-2
        The weight of the group-lasso penalty in the optimisation scheme.

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
    groups_: list of lists of int or None
        The explicit partition of the variables formed by the groups parameter if it was not None.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso
    Sparse GEMINI - Sparse GEMINI for joint discriminative clustering and feature selection
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso


    See Also
    --------
    SparseLinearModel: sparse logistic regression trained with any GEMINI
    SparseLinearMI: sparse logistic regression trained for clustering with the mutual information

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.sparse import SparseLinearMMD
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = SparseLinearMMD(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.7040618744
    """
    _parameter_constraints: dict = {
        **SparseLinearModel._parameter_constraints,
        "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
        "kernel_params": [dict, None],
        "ovo": [bool]
    }

    def __init__(self, n_clusters=3, groups=None, max_iter=1000, learning_rate=1e-3, kernel="linear", ovo=False,
                 alpha=1e-2, dynamic=False, solver="adam", batch_size=None, verbose=False, random_state=None,
                 kernel_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=None,
            groups=groups,
            max_iter=max_iter,
            learning_rate=learning_rate,
            dynamic=dynamic,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            alpha=alpha
        )
        self.ovo = ovo
        self.kernel = kernel
        self.kernel_params = kernel_params

    def get_gemini(self):
        return MMDGEMINI(ovo=self.ovo, kernel=self.kernel, kernel_params=self.kernel_params)


class SparseLinearMI(SparseLinearModel):
    """ This is the Sparse version of the logistic regression trained with mutual information for clustering.

    On top of the vanilla logistic regression model, this variation brings a group-lasso penalty constraint to ensure
    feature selection via a proximal gradient during training. The objective function is the mutual information.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    groups: list of arrays of various shapes, default=None
        If groups is set, it must describe a partition of the indices of variables. This will be used for performing
        variable selection with groups of features considered to represent one variable. This option can typically be
        used for one-hot-encoded variables. Variable indices that are not entered will be considered alone.
        For example, with 3 features, accepted values can be [[0],[1],[2]], [[0,1],[2]] or [[0,1]].

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    alpha: float, default=1e-2
        The weight of the group-lasso penalty in the optimisation scheme.

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
    groups_: list of lists of int or None
        The explicit partition of the variables formed by the groups parameter if it was not None.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso
    Sparse GEMINI - Sparse GEMINI for joint discriminative clustering and feature selection
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso
    Sparse MI logistic regression - Discriminative Clustering and Feature Selection for Brain MRI Segmentation
        Youyong Kong, Yue Deng, Qionghai Dai


    See Also
    --------
    SparseLinearModel: sparse logistic regression trained with any GEMINI
    SparseLinearMMD: sparse logistic regression trained for clustering with the MMD GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.sparse import SparseLinearMI
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = SparseLinearMI(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    0.5812412917
    """

    _parameter_constraints: dict = {
        **SparseLinearModel._parameter_constraints,
    }

    def __init__(self, n_clusters=3, groups=None, max_iter=1000, learning_rate=1e-3,
                 alpha=1e-2, solver="adam", batch_size=None, verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini="mi",
            groups=groups,
            max_iter=max_iter,
            dynamic=False,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            alpha=alpha
        )
