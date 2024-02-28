import functools
import itertools
from numbers import Real

import numpy as np
from scipy.sparse import csgraph
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval

from ._base_gemini import DiscriminativeModel
from ._constraints import constraint_params


def _check_structural_constraint(must_link, cannot_link):
    # First establish all connected components
    unique_indices = [p[0] for p in must_link] + [p[1] for p in must_link]
    unique_indices = list(set(unique_indices))

    connection_matrix = np.zeros((len(unique_indices), len(unique_indices)))
    for pair in must_link:
        i, j = unique_indices.index(pair[0]), unique_indices.index(pair[1])
        connection_matrix[i, j] = connection_matrix[j, i] = 1

    samples_to_explore = list(range(len(unique_indices)))
    while len(samples_to_explore) != 0:
        # Perform simple bfs algorithm to search all reachable nodes starting from a sample
        reacheable_nodes = csgraph.breadth_first_order(connection_matrix,
                                                       samples_to_explore[0],
                                                       directed=False,
                                                       return_predecessors=False)

        for node in reacheable_nodes:
            samples_to_explore.remove(node)

        for i, j in itertools.combinations(reacheable_nodes, r=2):

            for pair in cannot_link:
                pair_i, pair_j = pair
                if (i == pair_i and j == pair_j) or (i == pair_j and j == pair_i):
                    raise ValueError("Triangular contradiction in Must-link / Cannot-link constraints")


def _check_linking_constraint(must_link=None, cannot_link=None):
    if hasattr(must_link, "__len__") and len(must_link) == 0:
        must_link = None
    if hasattr(cannot_link, "__len__") and len(cannot_link) == 0:
        cannot_link = None
    if must_link is not None:
        must_link = check_array(must_link, ensure_2d=True, ensure_min_features=2, dtype=int,
                                input_name="Must-link constraint")
        # Check that we do not have any self-reference
        if np.any(must_link[:, 0] == must_link[:, 1]):
            raise ValueError("An element is necessary in the same cluster as itself, check constraints in must-link")
    else:
        must_link = []
    if cannot_link is not None:
        cannot_link = check_array(cannot_link, ensure_2d=True, ensure_min_features=2, dtype=int,
                                  input_name="Cannot-link constraint")

        # Check that we do not have any self-reference
        if np.any(cannot_link[:, 0] == cannot_link[:, 1]):
            raise ValueError("An element cannot be in a different cluster than itself, "
                             "check constraints in cannot-link")
    else:
        cannot_link = []

    if len(must_link) > 0 and len(cannot_link) > 0:
        # Check that we do not have any structural contradiction
        _check_structural_constraint(must_link, cannot_link)


@constraint_params({
    "gemini_model": [DiscriminativeModel],
    "must-link": ["array-like", None],
    "cannot-link": ["array-like", None],
    "factor": [Interval(Real, 0, None, closed="neither")]
})
def add_mlcl_constraint(gemini_model, must_link=None, cannot_link=None, factor=1.0):
    """
    Adds must-link and/or cannot-link constraints to a discriminative clustering model. The contraints are ensure by
    respectively minimising or maximising the :math:`\ell_2` norm between the prediction vectors. It is thus possible
    that not all constraints are fully satisfied.

    Parameters
    ----------
    gemini_model: MLP___, Linear___ or Categorical___
        A GemClus model that involves gemini maximisation with gradient descent.

    must_link: ndarray of shape (n_constraints, 2) or None, default=None
        The constraints of samples being together must be described by a list of pairs of indices
        [(i1,j1),..., (iN, jN)].
        If set to None, no must-link constraint is applied on the model.

    cannot_link: ndarray of shape (n_constraints, 2) or None, default=None
        The constraints of samples which must not be in the same cluster must be described by a list of pairs of indices
        [(i1,j1),..., (iN, jN)].
        If set to None, no cannot-link constraint is applied on the model.

    factor: float, default=1.0
        A weighting hyperparameter for the constraints in gradient descent.

    Returns
    -------
    The model gemini model with decorated gradient functions to satisfy must-link / cannot-link constraints.
    """

    if not issubclass(gemini_model.__class__, DiscriminativeModel):
        raise ValueError(f"The passed model does not inherit from the DiscriminativeModel class: "
                         f"{gemini_model.__class__}")

    _check_linking_constraint(must_link, cannot_link)

    if must_link is None:
        must_link = []
    if cannot_link is None:
        cannot_link = []

    # We start by decorating the _batchify method such that we remember the indices of the samples
    def decorate_batch(func):
        @functools.wraps(func)
        def disguise_batch(X, affinity_matrix=None, random_state=None):
            indices = np.arange(len(X))
            for subset, affinity_batch in func(indices, affinity_matrix, random_state):
                disguise_batch.indices = subset.tolist()
                yield X[subset], affinity_batch

        disguise_batch.indices = []
        return disguise_batch

    gemini_model._batchify = decorate_batch(gemini_model._batchify)

    # Now we can decorate the gradient computation by intercepting it and adding constraints
    # relative to must-link/cannot-link using known batch indices
    def decorate_grads(func):
        @functools.wraps(func)
        def intercept_grads(X, y_pred, gradient):
            # Retrieve the indices of the last call
            last_indices = gemini_model._batchify.indices
            for (i, j) in cannot_link:
                if i in last_indices and j in last_indices:
                    idx0, idx1 = last_indices.index(i), last_indices.index(j)
                    gradient[idx0] += factor * (y_pred[idx0] - y_pred[idx1])
                    gradient[idx1] += factor * (y_pred[idx1] - y_pred[idx0])
            for (i, j) in must_link:
                if i in last_indices and j in last_indices:
                    idx0, idx1 = last_indices.index(i), last_indices.index(j)
                    gradient[idx0] -= factor * (y_pred[idx0] - y_pred[idx1])
                    gradient[idx1] -= factor * (y_pred[idx1] - y_pred[idx0])
            return func(X, y_pred, gradient)

        return intercept_grads

    gemini_model._compute_grads = decorate_grads(gemini_model._compute_grads)

    return gemini_model
