import pytest
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.estimator_checks import check_clustering, check_methods_sample_order_invariance, \
    check_methods_subset_invariance

from ..linear import LinearWasserstein, LinearMMD, RIM, LinearModel
from ..mlp import MLPMMD, MLPWasserstein, MLPModel
from ..nonparametric import CategoricalWasserstein, CategoricalMMD, CategoricalModel
from ..sparse import SparseMLPMMD, SparseLinearMMD, SparseLinearMI, SparseLinearModel, SparseMLPModel


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X


@pytest.mark.parametrize(
    "clf",
    [LinearMMD(), MLPMMD(), LinearWasserstein(), MLPWasserstein(), SparseLinearMMD(), SparseMLPMMD(), RIM(),
     CategoricalMMD(), CategoricalWasserstein(), SparseLinearMI(), LinearModel(), MLPModel(), SparseMLPModel(),
     SparseLinearModel(), CategoricalModel()]
)
def test_default_clf_init(clf):
    assert clf.learning_rate == 1e-3
    assert clf.n_clusters == 3
    assert clf.max_iter == 1000
    assert clf.random_state is None
    assert clf.solver == "adam"


@pytest.mark.parametrize(
    "clf",
    [LinearModel, LinearMMD, LinearWasserstein, RIM]
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
    [CategoricalModel, CategoricalMMD, CategoricalWasserstein]
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
    [MLPModel, MLPMMD, MLPWasserstein, SparseMLPMMD]
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
    [SparseMLPModel, SparseMLPMMD]
)
def test_all_sparse_lassonet_attributes(clf, data):
    clf = clf(max_iter=1)
    assert clf.alpha == 1e-2
    assert clf.M == 10
    clf.fit(data)
    assert hasattr(clf, "W_skip_")


@pytest.mark.parametrize(
    "clf",
    [SparseLinearModel, SparseLinearMMD, SparseLinearMI]
)
def test_all_sparse_linear_attributes(clf, data):
    clf = clf(max_iter=1)
    assert clf.alpha == 1e-2


@pytest.mark.parametrize(
    "clf",
    [LinearMMD(), MLPMMD(), SparseMLPMMD(), SparseLinearMMD(), CategoricalMMD()]
)
def test_default_mmd(clf):
    assert clf.kernel == "linear"
    assert not clf.ovo


@pytest.mark.parametrize(
    "clf",
    [LinearWasserstein(), MLPWasserstein(), CategoricalWasserstein()]
)
def test_default_wasserstein(clf):
    assert clf.metric == "euclidean"
    assert not clf.ovo


@pytest.mark.parametrize(
    "estimator",
    [MLPMMD(max_iter=5), LinearMMD(max_iter=5), SparseMLPMMD(max_iter=5), SparseLinearMMD(max_iter=5),
     MLPWasserstein(max_iter=5), LinearWasserstein(max_iter=5), RIM(max_iter=5), SparseLinearMI(max_iter=5),
     LinearModel(max_iter=5), MLPModel(max_iter=5), SparseLinearModel(max_iter=5), SparseMLPModel(max_iter=5)]
)
def test_all_estimators(estimator):
    check_iterator = check_estimator(estimator, generate_only=True)
    for clf, check in check_iterator:
        if check.func == check_clustering:
            # The check clustering function tests if we output as many clusters as promised,
            # But since GEMINI may have fewer clusters, this test will never be satisfied
            continue
        check(clf)


@pytest.mark.parametrize(
    "estimator",
    [CategoricalModel(max_iter=5), CategoricalMMD(max_iter=5), CategoricalWasserstein(max_iter=5)]
)
def test_categorical_estimators(estimator):
    check_iterator = check_estimator(estimator, generate_only=True)
    for clf, check in check_iterator:
        if check.func in [check_clustering, check_methods_sample_order_invariance, check_methods_subset_invariance]:
            continue
        check(clf)


@pytest.mark.parametrize(
    "estimator",
    [MLPModel, MLPMMD, LinearModel, LinearMMD, SparseMLPMMD, SparseLinearMMD, SparseLinearMI, MLPWasserstein,
     LinearWasserstein, RIM]
)
@pytest.mark.parametrize(
    "batch_size",
    [10, 50, None]
)
def test_batch_size(estimator, batch_size, data):
    clf = estimator(max_iter=5, batch_size=batch_size)
    clf.fit(data)
    assert hasattr(clf, "labels_")
