import pytest
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator, check_fit2d_predict1d
from sklearn.utils.estimator_checks import check_clustering, check_methods_sample_order_invariance, \
    check_methods_subset_invariance

from ..linear import LinearWasserstein, LinearMMD, RIM, LinearModel, KernelRIM
from ..mlp import MLPMMD, MLPWasserstein, MLPModel
from ..nonparametric import CategoricalWasserstein, CategoricalMMD, CategoricalModel
from ..sparse import SparseMLPMMD, SparseLinearMMD, SparseLinearMI, SparseLinearModel, SparseMLPModel

all_models = [LinearMMD, MLPMMD, LinearWasserstein, MLPWasserstein, SparseLinearMMD, SparseMLPMMD, RIM,
              CategoricalMMD, CategoricalWasserstein, SparseLinearMI, LinearModel, MLPModel, SparseMLPModel,
              SparseLinearModel, CategoricalModel, KernelRIM]

@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X


@pytest.mark.parametrize(
    "clf",
    all_models
)
def test_default_clf_init(clf):
    clf = clf()
    assert clf.learning_rate == 1e-3
    assert clf.n_clusters == 3
    assert clf.max_iter == 1000
    assert clf.random_state is None
    assert clf.solver == "adam"


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "Linear" in x.__name__] + [RIM, KernelRIM]
)
def test_all_linear_attributes(clf, data):
    clf = clf(max_iter=1)
    clf.fit(data)
    assert hasattr(clf, 'W_')
    assert hasattr(clf, 'b_')
    assert hasattr(clf, 'optimiser_')
    assert hasattr(clf, 'labels_')
    assert hasattr(clf, 'n_iter_')


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "Categorical" in x.__name__]
)
def test_all_categorical_attributes(clf, data):
    clf = clf(max_iter=1)
    clf.fit(data)
    assert hasattr(clf, 'logits_')
    assert hasattr(clf, 'optimiser_')
    assert hasattr(clf, 'labels_')
    assert hasattr(clf, 'n_iter_')


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "MLP" in x.__name__]
)
def test_all_mlp_attributes(clf, data):
    clf = clf(max_iter=1)
    assert clf.n_hidden_dim == 20
    clf.fit(data)
    assert hasattr(clf, 'W1_')
    assert hasattr(clf, 'W2_')
    assert hasattr(clf, 'b1_')
    assert hasattr(clf, 'b2_')
    assert hasattr(clf, 'optimiser_')
    assert hasattr(clf, 'labels_')
    assert hasattr(clf, 'n_iter_')
    assert hasattr(clf, 'H_')


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "SparseMLP" in x.__name__]
)
def test_all_sparse_lassonet_attributes(clf, data):
    clf = clf(max_iter=1)
    assert clf.alpha == 1e-2
    assert clf.M == 10
    clf.fit(data)
    assert hasattr(clf, "W_skip_")


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "SparseLinear" in x.__name__]
)
def test_all_sparse_linear_attributes(clf, data):
    clf = clf(max_iter=1)
    assert clf.alpha == 1e-2


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "MMD" in x.__name__]
)
def test_default_mmd(clf):
    clf = clf()
    assert clf.kernel == "linear"
    assert not clf.ovo


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "Wasserstein" in x.__name__]
)
def test_default_wasserstein(clf):
    clf = clf()
    assert clf.metric == "euclidean"
    assert not clf.ovo


@pytest.mark.parametrize(
    "clf",
    all_models
)
def test_all_estimators(clf):
    estimator = clf(max_iter=5)
    check_iterator = check_estimator(estimator, generate_only=True)
    for clf, check in check_iterator:
        if check.func == check_clustering:
            # The check clustering function tests if we output as many clusters as promised,
            # But since GEMINI may have fewer clusters, this test will never be satisfied
            continue
        if check.func in [check_methods_sample_order_invariance, check_methods_subset_invariance] \
                and "Categorical" in type(estimator).__name__:
            continue
        check(clf)


@pytest.mark.parametrize(
    "clf",
    [x for x in all_models if "Categorical" not in x.__name__ and x.__name__ != "KernelRIM"]
)
@pytest.mark.parametrize(
    "batch_size",
    [10, 50, None]
)
def test_batch_size(clf, batch_size, data):
    clf = clf(max_iter=5, batch_size=batch_size)
    clf.fit(data)
    assert hasattr(clf, "labels_")
