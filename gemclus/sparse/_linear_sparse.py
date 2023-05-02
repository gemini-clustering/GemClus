from .._base_gemini import _BaseMMD
from ..sparse._base_sparse import _SparseLinearGEMINI


class SparseLinearMMD(_SparseLinearGEMINI, _BaseMMD):
    """ This is the Sparse version of the LinearMMD clustering model.

    On top of the vanilla Linear GEMINI model, this variation brings a group-lasso penalty constraint to ensure
    feature selection via a proximal gradient during training.

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
    LassoNet architecture - LassoNet: A Neural Network with Feature Sparsity.
        Lemhadri, I., Ruan, F., Abraham, L., & Tibshirani, R.
    Sparse GEMINI - Sparse GEMINI for joint discriminative clustering and feature selection
        Louis Ohl, Pierre-Alexandre Mattei, Charles Bouveyron, Mickaël Leclercq,
        Arnaud Droit, Frederic Preciosio


    See Also
    --------
    SparseMLPMMD: sparse two-layer neural network trained for clustering with the MMD GEMINI

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
    1.6940342321220005
    """
    _parameter_constraints: dict = {
        **_SparseLinearGEMINI._parameter_constraints,
        **_BaseMMD._parameter_constraints,
    }

    def __init__(self, n_clusters=3, groups=None, max_iter=1000, learning_rate=1e-3, kernel="linear", ovo=False,
                 alpha=1e-2, solver="adam", batch_size=None, verbose=False, random_state=None):
        _SparseLinearGEMINI.__init__(self,
                                     n_clusters=n_clusters,
                                     groups=groups,
                                     max_iter=max_iter,
                                     learning_rate=learning_rate,
                                     solver=solver,
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     random_state=random_state,
                                     alpha=alpha
                                     )
        _BaseMMD.__init__(self,
                          n_clusters=n_clusters,
                          max_iter=max_iter,
                          learning_rate=learning_rate,
                          solver=solver,
                          kernel=kernel,
                          ovo=ovo,
                          batch_size=batch_size,
                          verbose=verbose,
                          random_state=random_state,
                          )
