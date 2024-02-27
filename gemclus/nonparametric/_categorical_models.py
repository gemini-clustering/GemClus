from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.extmath import softmax

from .._base_gemini import DiscriminativeModel
from ..gemini import MMDGEMINI, WassersteinGEMINI


class CategoricalModel(DiscriminativeModel):
    """ The CategoricalModel is a nonparametric model where each sample is directly assign a probability vector of
    as conditional clustering distribution. Consequently, the parameters do not depend on the value of :math:`$x$`.

    .. math::
        p(y=k|x_i) = \\theta_{ki}

    Contrarily to other models, the categorical distribution can not be used for clustering samples that were not
    part of the training set and does not support batching as well.

    The model optimises the parameters to maximise any GEMINI.

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
    logits_: ndarray of shape (n_samples, n_clusters)
        The logit of the cluster membership of each sample.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    CategoricalMMD: nonparametric model tailored for the MMD GEMINI
    CategoricalWasserstein: nonparametric model tailored for the Wasserstein GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.nonparametric import CategoricalModel
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = CategoricalModel(gemini="mi",random_state=0).fit(X)
    >>> clf.score(X)
    0.6577369504
    """
    _parameter_constraints: dict = {
        **DiscriminativeModel._parameter_constraints,
    }

    def __init__(self, n_clusters=3, gemini="mmd_ova", max_iter=1000, learning_rate=1e-3, solver="adam",
                 verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=gemini,
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
        yield X, affinity_matrix


class CategoricalMMD(CategoricalModel):
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

    kernel_params: dict, default=None
        A dictionary of keyword arguments to pass to the chosen kernel function.

    Attributes
    ----------
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    logits_: ndarray of shape (n_samples, n_clusters)
        The logit of the cluster membership of each sample.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    CategoricalModel: nonparametric model tailored for any generic GEMINI
    CategoricalWasserstein: nonparametric model tailored for the Wasserstein GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.nonparametric import CategoricalMMD
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = CategoricalMMD(random_state=0).fit(X)
    >>> clf.score(X)
    1.2117267518
    """
    _parameter_constraints: dict = {
        **CategoricalModel._parameter_constraints,
        "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
        "kernel_params": [dict, None],
        "ovo": [bool]
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, solver="adam", kernel="linear", ovo=False,
                 verbose=False, random_state=None, kernel_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=None,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            verbose=verbose,
            random_state=random_state
        )
        self.ovo = ovo
        self.kernel = kernel
        self.kernel_params = kernel_params

    def get_gemini(self):
        return MMDGEMINI(ovo=self.ovo, kernel=self.kernel, kernel_params=self.kernel_params)


class CategoricalWasserstein(CategoricalModel):
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

    metric_params: dict, default=None
        A dictionary of keyword arguments to pass to the chosen metric function.

    Attributes
    ----------
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    logits_: ndarray of shape (n_samples, n_clusters)
        The logit of the cluster membership of each sample.

    References
    ----------
    GEMINI - Generalised Mutual Information for Discriminative Clustering
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Warith Harchaoui, Mickaël Leclercq,
        Arnaud Droit, Frederic Precioso

    See Also
    --------
    CategoricalModel: nonparametric model tailored for any generic GEMINI
    CategoricalMMD: nonparametric model tailored for the MMD GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.nonparametric import CategoricalWasserstein
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = CategoricalWasserstein(random_state=0).fit(X)
    >>> clf.score(X)
    1.3555482569
    """
    _parameter_constraints: dict = {
        **CategoricalModel._parameter_constraints,
        "metric": [StrOptions(set(list(PAIRWISE_DISTANCE_FUNCTIONS) + ["precomputed"])), callable],
        "metric_params": [dict, None],
        "ovo": [bool],
    }

    def __init__(self, n_clusters=3, max_iter=1000, learning_rate=1e-3, metric="euclidean", ovo=False,
                 solver="adam", verbose=False, random_state=None, metric_params=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=None,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            verbose=verbose,
            random_state=random_state,
        )
        self.ovo = ovo
        self.metric = metric
        self.metric_params = metric_params

    def get_gemini(self):
        return WassersteinGEMINI(ovo=self.ovo, metric=self.metric, metric_params=self.metric_params)
