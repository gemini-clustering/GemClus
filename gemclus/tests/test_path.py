import numpy as np
import pytest
from sklearn.datasets import make_blobs

from ..sparse import SparseMLPMMD, SparseLinearMMD


@pytest.fixture
def data():
    np.random.seed(0)
    X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.1, random_state=0)
    X = np.concatenate([X, np.random.normal(size=(100, 8))], axis=1)
    return X


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD]
)
def test_sparse_stability(clf_class, data):
    clf = clf_class(random_state=0)

    best_weights, geminis, penalties, alphas, n_features = clf.path(data, restore_best_weights=False)

    assert len(geminis) == len(alphas)
    assert len(geminis) == len(n_features)
    assert len(best_weights) == len(clf._get_weights())

    assert n_features[-1] <= 2

    clf = clf_class(random_state=0)

    best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data, restore_best_weights=True)

    for i in range(len(best_weights_v2)):
        assert np.allclose(clf._get_weights()[i], best_weights_v2[i])
        assert np.allclose(best_weights[i], best_weights_v2[i])
    assert np.allclose(geminis, geminis_v2)
    assert np.allclose(alphas, alphas_v2)
    assert np.allclose(n_features, n_features_v2)

    gemini_score = clf.score(data)

    assert gemini_score >= 0.9 * max(geminis)


def test_weights_coherence(data):
    clf = SparseMLPMMD(random_state=0)

    best_weights, geminis, penalties, alphas, n_features = clf.path(data, restore_best_weights=False)

    assert n_features[-1] <= 2
    assert n_features[-1] == (np.linalg.norm(clf.W1_, ord=2, axis=1) != 0).sum()
    assert n_features[-1] == (np.linalg.norm(clf.W_skip_, ord=2, axis=1) != 0).sum()


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD]
)
def test_erroneous_multiplier(clf_class, data):
    clf = clf_class(random_state=0)

    best_weights_v1, geminis_v1, penalties_v1, alphas_v1, n_features_v1 = clf.path(data, restore_best_weights=False)

    clf = clf_class(random_state=0)

    best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data, alpha_multiplier=-1,
                                                                                   restore_best_weights=False)

    assert np.allclose(np.array(geminis_v1), np.array(geminis_v2))
    assert np.allclose(np.array(alphas_v1), np.array(alphas_v2))
    assert np.allclose(np.array(n_features_v1), np.array(n_features_v2))

    for i in range(len(best_weights_v1)):
        assert np.allclose(best_weights_v1[i], best_weights_v2[i])


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD]
)
def test_erroneous_min_features(clf_class, data):
    clf = clf_class(random_state=0)

    best_weights_v1, geminis_v1, penalties_v1, alphas_v1, n_features_v1 = clf.path(data, restore_best_weights=False)

    clf = clf_class(random_state=0)

    best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data, min_features=0,
                                                                                   restore_best_weights=False)

    for i in range(len(best_weights_v1)):
        assert np.allclose(best_weights_v1[i], best_weights_v2[i])

    assert np.allclose(np.array(geminis_v1), np.array(geminis_v2))
    assert np.allclose(np.array(alphas_v1), np.array(alphas_v2))
    assert np.allclose(np.array(n_features_v1), np.array(n_features_v2))


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD]
)
def test_excessive_threshold(clf_class, data):
    clf = clf_class(random_state=0)

    best_weights_v1, geminis_v1, penalties_v1, alphas_v1, n_features_v1 = clf.path(data, restore_best_weights=False)

    clf = clf_class(random_state=0)

    best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data, keep_threshold=2,
                                                                                   restore_best_weights=False)

    for i in range(len(best_weights_v1)):
        assert np.allclose(best_weights_v1[i], best_weights_v2[i])

    assert np.allclose(np.array(geminis_v1), np.array(geminis_v2))
    assert np.allclose(np.array(alphas_v1), np.array(alphas_v2))
    assert np.allclose(np.array(n_features_v1), np.array(n_features_v2))


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD]
)
def test_erroneous_threshold(clf_class, data):
    clf = clf_class(random_state=0)

    best_weights_v1, geminis_v1, penalties_v1, alphas_v1, n_features_v1 = clf.path(data, restore_best_weights=False)

    clf = clf_class(random_state=0)
    best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data, keep_threshold=-1,
                                                                                   restore_best_weights=False)

    for i in range(len(best_weights_v1)):
        assert np.allclose(best_weights_v1[i], best_weights_v2[i])

    assert np.allclose(np.array(geminis_v1), np.array(geminis_v2))
    assert np.allclose(np.array(alphas_v1), np.array(alphas_v2))
    assert np.allclose(np.array(n_features_v1), np.array(n_features_v2))

@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD]
)
def test_batch_path(clf_class, data):
    clf = clf_class(batch_size=50, random_state=0, max_iter=100)

    best_weights_v1, geminis_v1, penalties_v1, alphas_v1, n_features_v1 = clf.path(data, restore_best_weights=False,
                                                                                   min_features=data.shape[1]-1)
