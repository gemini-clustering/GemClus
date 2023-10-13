import numpy as np
import pytest

from sklearn import datasets

from gemclus.nonparametric import CategoricalMMD, CategoricalWasserstein
from gemclus.linear import LinearMMD, LinearWasserstein
from gemclus.sparse import SparseMLPMMD, SparseLinearMMD
from gemclus.mlp import MLPMMD, MLPWasserstein
from ..mlcl import add_mlcl_constraint


@pytest.mark.parametrize(
    "model_class",
    [CategoricalMMD, CategoricalWasserstein]
)
def test_constraints_satisfaction(model_class):
    # Generate some data (interlacing moons)
    X, y = datasets.make_moons(n_samples=100, noise=0.08)

    # Create random constraints to cluster differently samples from the previous result
    I0, I1 = np.nonzero(1 - y)[0], np.nonzero(y)[0]

    # Must link
    np.random.seed(0)
    ml0 = [I0[np.random.permutation(len(I0))[:2]] for i in range(5)]
    ml1 = [I1[np.random.permutation(len(I1))[:2]] for i in range(5)]
    must_link = ml0 + ml1
    # Cannot link
    cannot_link = [np.array([I0[np.random.randint(0, len(I0))], I1[np.random.randint(0, len(I1))]]) for i in range(10)]

    constrained_model = add_mlcl_constraint(model_class(n_clusters=2, random_state=0), must_link, cannot_link)
    constrained_model.fit(X)
    y_pred_constraint = constrained_model.predict(X)

    for pair in must_link:
        assert y_pred_constraint[pair[0]] == y_pred_constraint[pair[1]], "ML violation for free model"
    for pair in cannot_link:
        assert y_pred_constraint[pair[0]] != y_pred_constraint[pair[1]], "CL violation for free model"


@pytest.mark.parametrize(
    "bad_constraint",
    [[(0,)], 3, (0, 2), [2], np.array([[0], [1]])]
)
@pytest.mark.parametrize(
    "model_class",
    [LinearMMD, CategoricalMMD]
)
def test_wrong_constraints(bad_constraint, model_class):
    with pytest.raises(ValueError) as excinfo:
        model = add_mlcl_constraint(model_class(n_clusters=2, max_iter=5),
                                    must_link=bad_constraint,
                                    cannot_link=None)

    with pytest.raises(ValueError) as excinfo:
        model = add_mlcl_constraint(model_class(n_clusters=2, max_iter=5),
                                    must_link=None,
                                    cannot_link=bad_constraint)

    with pytest.raises(ValueError) as excinfo:
        model = add_mlcl_constraint(model_class(n_clusters=2, max_iter=5),
                                    must_link=bad_constraint,
                                    cannot_link=bad_constraint)


@pytest.mark.parametrize(
    "good_constraint",
    [[(0, 1)], None, [], [(0, 1), (0, 2)], np.array([]), np.array([[0, 1]]), np.array([[0, 1], [1, 2]])]
)
@pytest.mark.parametrize(
    "model_class",
    [LinearMMD, CategoricalMMD]
)
def test_good_constraints(good_constraint, model_class):
    add_mlcl_constraint(model_class(n_clusters=2, max_iter=5),
                        must_link=good_constraint,
                                cannot_link=None)
    add_mlcl_constraint(model_class(n_clusters=2, max_iter=5),
                        must_link=None,
                        cannot_link=good_constraint)


@pytest.mark.parametrize(
    "model_class",
    [CategoricalMMD, CategoricalWasserstein, LinearMMD, LinearWasserstein, MLPWasserstein, MLPMMD, SparseLinearMMD,
     SparseMLPMMD]
)
def test_name_masking(model_class):
    regular_model = model_class()

    constrained_model = add_mlcl_constraint(regular_model, [(0, 1)], [(1, 2)])

    assert regular_model._compute_grads.__name__ == constrained_model._compute_grads.__name__
    if regular_model._compute_grads.__doc__ is not None:
        assert regular_model._compute_grads.__doc__ == constrained_model._compute_grads.__doc__
    assert regular_model._compute_grads.__class__ == constrained_model._compute_grads.__class__

    assert regular_model._batchify.__name__ == constrained_model._batchify.__name__
    if regular_model._batchify.__doc__ is not None:
        assert regular_model._batchify.__doc__ == constrained_model._batchify.__doc__
    assert regular_model._batchify.__class__ == constrained_model._batchify.__class__


def test_conflictual_constraints():
    with pytest.raises(AssertionError) as excinfo:
        # Items in tuple should always differ
        model = add_mlcl_constraint(LinearMMD(n_clusters=2),
                                    must_link=[(0, 0)],
                                    cannot_link=None)
    assert str(excinfo.value) == (f"An element is necessary in the same cluster as itself, check "
                                  f"constraints in must-link")

    with pytest.raises(AssertionError) as excinfo:
        # Items in tuple should always differ
        model = add_mlcl_constraint(LinearMMD(n_clusters=2),
                                    must_link=None,
                                    cannot_link=[(0, 0)])
    assert str(excinfo.value) == (f"An element cannot be in a different cluster than itself, "
                                  f"check constraints in cannot-link")

    with pytest.raises(AssertionError) as excinfo:
        # Here, the constraint are contradictory
        model = add_mlcl_constraint(LinearMMD(n_clusters=2),
                                    must_link=[(0, 1), (1, 2)],
                                    cannot_link=[(1, 2)])
    assert str(excinfo.value) == "Triangular contradiction in Must-link / Cannot-link constraints"
