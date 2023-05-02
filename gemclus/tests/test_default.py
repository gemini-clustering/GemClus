import pytest
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.estimator_checks import check_clustering

from ..linear import LinearWasserstein, LinearMMD, RIM
from ..mlp import MLPMMD, MLPWasserstein
from ..sparse import SparseMLPMMD, SparseLinearMMD


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X


@pytest.mark.parametrize(
    "clf",
    [LinearMMD(), MLPMMD(), LinearWasserstein(), MLPWasserstein(), SparseLinearMMD(), SparseMLPMMD(), RIM()]
)
def test_default_clf_init(clf):
    assert clf.learning_rate == 1e-3
    assert clf.n_clusters == 3
    assert clf.max_iter == 1000
    assert clf.random_state is None
    assert clf.solver == "adam"


@pytest.mark.parametrize(
    "clf",
    [LinearMMD, LinearWasserstein, RIM]
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
    [MLPMMD, MLPWasserstein, SparseMLPMMD]
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
    [SparseMLPMMD]
)
def test_all_sparse_lassonet_attributes(clf, data):
    clf = clf(max_iter=1)
    assert clf.alpha == 1e-2
    assert clf.M == 10
    clf.fit(data)
    assert hasattr(clf, "W_skip_")


@pytest.mark.parametrize(
    "clf",
    [SparseLinearMMD]
)
def test_all_sparse_linear_attributes(clf, data):
    clf = clf(max_iter=1)
    assert clf.alpha == 1e-2


@pytest.mark.parametrize(
    "clf",
    [LinearMMD(), MLPMMD(), SparseMLPMMD(), SparseLinearMMD()]
)
def test_default_mmd(clf):
    assert clf.kernel == "linear"
    assert not clf.ovo


@pytest.mark.parametrize(
    "clf",
    [LinearWasserstein(), MLPWasserstein()]
)
def test_default_wasserstein(clf):
    assert clf.metric == "euclidean"
    assert not clf.ovo


@pytest.mark.parametrize(
    "estimator",
    [MLPMMD(max_iter=5), LinearMMD(max_iter=5), SparseMLPMMD(max_iter=5), SparseLinearMMD(max_iter=5),
     MLPWasserstein(max_iter=5), LinearWasserstein(max_iter=5), RIM(max_iter=5)]
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
    [MLPMMD, LinearMMD, SparseMLPMMD, SparseLinearMMD, MLPWasserstein, LinearWasserstein, RIM]
)
@pytest.mark.parametrize(
    "batch_size",
    [10,50,None]
)
def test_batch_size(estimator, batch_size, data):
    clf = estimator(max_iter=5, batch_size=batch_size)
    clf.fit(data)
    assert hasattr(clf, "labels_")
