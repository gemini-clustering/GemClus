from .._base_gemini import _BaseMMD
from ..sparse._base_sparse import _SparseMLPGEMINI


class SparseMLPMMD(_SparseMLPGEMINI, _BaseMMD):
    """ This is the Sparse Version of the MLP MMD model.

    On top of the vanilla MLP GEMINI model, this variation brings a skip connection from the data to the cluster
    output. This skip connection ensures a sparsity constraint through a group-lasso penalty and a proximal gradient
    that eliminates input features as well in the first layer of the MLP.

    This architecture is inspired from LassoNet by Lemhadri et al (2021).

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    groups: list of arrays of various shapes, default=None
        if groups is set, it must describe a partition of the indices of variables. This will be used for performing
        variable selection with groups of features considered to represent one variables. This option can typically be
        used for one-hot-encoded variables.

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

    alpha: float, default=1e-2
        The weight of the group-lasso penalty in the optimisation scheme.

    M: float, default=10 The hierarchy coefficient that controls the relative strength between the group-lasso
        penalty of the skip connection and the sparsity of the first layer of the MLP.

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
    W_skip_: ndarray of shape (n_features, n_clusters)
        The linear weights of the skip connection
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
    LassoNet architecture - LassoNet: A Neural Network with Feature Sparsity.
        Lemhadri, I., Ruan, F., Abraham, L., & Tibshirani, R.
    Sparse GEMINI - Sparse GEMINI for joint discriminative clustering and feature selection
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Mickaël Leclercq,
        Arnaud Droit, Frederic Preciosio


    See Also
    --------
    SparseLinearMMD: sparse logistic regression trained for clustering with the MMD GEMINI

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gemclus.sparse import SparseMLPMMD
    >>> X,y=load_iris(return_X_y=True)
    >>> clf = SparseMLPMMD(random_state=0).fit(X)
    >>> clf.predict(X[:2,:])
    array([0, 0])
    >>> clf.predict_proba(X[:2,:]).shape
    (2, 3)
    >>> clf.score(X)
    1.7664211836410726
    """
    _parameter_constraints: dict = {
        **_SparseMLPGEMINI._parameter_constraints,
    }

    def __init__(self, n_clusters=3, groups=None, max_iter=1000, learning_rate=1e-3, n_hidden_dim=20, kernel="linear",
                 M=10, batch_size=None, alpha=1e-2, ovo=False, solver="adam", verbose=False, random_state=None):
        _SparseMLPGEMINI.__init__(
            self,
            n_clusters=n_clusters,
            groups=groups,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_hidden_dim=n_hidden_dim,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            M=M,
            alpha=alpha
        )
        _BaseMMD.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            kernel=kernel,
            ovo=ovo,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
        )
