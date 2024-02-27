import numpy as np
import pytest
from sklearn.datasets import make_blobs

import gemclus.gemini
from ..sparse import SparseMLPMMD, SparseLinearMMD, SparseLinearMI, SparseLinearModel, SparseMLPModel

sparse_models = [SparseMLPMMD, SparseLinearMMD, SparseLinearMI, SparseLinearModel, SparseMLPModel]


@pytest.fixture
def data():
    np.random.seed(0)
    X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.1, random_state=0)
    X = np.concatenate([X, np.random.normal(size=(100, 5))], axis=1)
    return X


@pytest.mark.parametrize(
    "clf_class",
    sparse_models
)
def test_sparse_stability(clf_class, data):
    min_features = data.shape[-1] - 2
    clf = clf_class(random_state=0)

    best_weights, geminis, penalties, alphas, n_features = clf.path(data, min_features=min_features,
                                                                    restore_best_weights=False)

    assert len(geminis) == len(alphas)
    assert len(geminis) == len(n_features)
    assert len(best_weights) == len(clf._get_weights())

    assert n_features[-1] <= min_features

    clf = clf_class(random_state=0)

    best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data, min_features=min_features,
                                                                                   restore_best_weights=True)

    for i in range(len(best_weights_v2)):
        assert np.allclose(clf._get_weights()[i], best_weights_v2[i])
        assert np.allclose(best_weights[i], best_weights_v2[i])
    assert np.allclose(geminis, geminis_v2)
    assert np.allclose(alphas, alphas_v2)
    assert np.allclose(n_features, n_features_v2)

    gemini_score = clf.score(data)

    # Locate the scores when using all features
    full_feature_models = np.where(np.array(n_features) == data.shape[1])[0]

    assert gemini_score >= 0.9 * max(np.array(geminis)[full_feature_models])


def test_weights_coherence(data):
    clf = SparseMLPMMD(random_state=0, alpha=10)

    best_weights, geminis, penalties, alphas, n_features = clf.path(data, restore_best_weights=False)

    assert n_features[-1] <= 2
    assert n_features[-1] == (np.linalg.norm(clf.W1_, ord=2, axis=1) != 0).sum()
    assert n_features[-1] == (np.linalg.norm(clf.W_skip_, ord=2, axis=1) != 0).sum()


@pytest.mark.parametrize(
    "clf_class",
    sparse_models
)
def test_erroneous_multiplier(clf_class, data):
    clf = clf_class(random_state=0, alpha=10)

    best_weights_v1, geminis_v1, penalties_v1, alphas_v1, n_features_v1 = clf.path(data,
                                                                                   min_features=data.shape[-1] - 1,
                                                                                   restore_best_weights=False)

    clf = clf_class(random_state=0, alpha=10)

    with pytest.warns(UserWarning):
        best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data,
                                                                                       min_features=data.shape[-1] - 1,
                                                                                       alpha_multiplier=-1,
                                                                                       restore_best_weights=False)

    assert np.allclose(np.array(geminis_v1), np.array(geminis_v2))
    assert np.allclose(np.array(alphas_v1), np.array(alphas_v2))
    assert np.allclose(np.array(n_features_v1), np.array(n_features_v2))

    for i in range(len(best_weights_v1)):
        assert np.allclose(best_weights_v1[i], best_weights_v2[i])


@pytest.mark.parametrize(
    "clf_class",
    sparse_models
)
def test_erroneous_min_features(clf_class, data):
    clf = clf_class(random_state=0)

    best_weights_v1, geminis_v1, penalties_v1, alphas_v1, n_features_v1 = clf.path(data, restore_best_weights=False)

    clf = clf_class(random_state=0)

    with pytest.warns(UserWarning):
        best_weights_v2, geminis_v2, penalties_v2, alphas_v2, n_features_v2 = clf.path(data, min_features=0,
                                                                                       restore_best_weights=False)

    for i in range(len(best_weights_v1)):
        assert np.allclose(best_weights_v1[i], best_weights_v2[i])

    assert np.allclose(np.array(geminis_v1), np.array(geminis_v2))
    assert np.allclose(np.array(alphas_v1), np.array(alphas_v2))
    assert np.allclose(np.array(n_features_v1), np.array(n_features_v2))


@pytest.mark.parametrize(
    "clf_class",
    sparse_models
)
@pytest.mark.parametrize(
    "threshold",
    [-1, 2]
)
def test_erroneous_threshold(clf_class, data, threshold):
    clf = clf_class(random_state=0, alpha=10)

    with pytest.warns(UserWarning):
        best_weights, geminis, penalties, alphas, n_features = clf.path(data,
                                                                        min_features=data.shape[-1] - 1,
                                                                        keep_threshold=threshold,
                                                                        restore_best_weights=False)

    assert max(geminis) >= clf.score(data) >= 0.9 * max(geminis)


@pytest.mark.parametrize(
    "clf_class",
    sparse_models
)
def test_batch_path(clf_class, data):
    clf = clf_class(batch_size=50, random_state=0, max_iter=100)

    clf.path(data, restore_best_weights=False, min_features=data.shape[1] - 1)


@pytest.mark.parametrize(
    "groups",
    [[np.array([0]), np.array([1, 2])], [[0], [1, 2]], [np.array([i]) for i in range(7)], [[i] for i in range(7)]]
)
@pytest.mark.parametrize(
    "clf_class",
    sparse_models
)
def test_good_groups(clf_class, groups, data):
    clf = clf_class(batch_size=50, random_state=0, max_iter=1, groups=groups)
    with pytest.warns(UserWarning):
        clf.path(data, restore_best_weights=False, min_features=data.shape[1])


@pytest.mark.parametrize(
    "groups",
    [[[-1]], [[0], [0]], [[0], [0, 1]], [[7]], [[0, 0], [2, 3, 4, 5, 6]]]
)
@pytest.mark.parametrize(
    "clf_class",
    sparse_models
)
def test_bad_groups(clf_class, groups, data):
    clf = clf_class(batch_size=50, random_state=0, max_iter=100, groups=groups)

    with pytest.raises(ValueError):
        clf.path(data, restore_best_weights=False, min_features=data.shape[1] - 1)


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD, SparseLinearModel, SparseMLPModel]
)
def test_dynamic(clf_class, data):
    clf = clf_class(batch_size=50, random_state=0, max_iter=100, dynamic=True)

    clf.path(data, restore_best_weights=False, min_features=data.shape[1] - 3)

    print(clf.get_selection())

    assert clf.dynamic


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPMMD, SparseLinearMMD]
)
def test_custom_metric_mmd(clf_class, data):
    clf = clf_class(batch_size=50, random_state=0, max_iter=100, kernel="precomputed", alpha=10)

    pre_computed_affinity = data @ data.T + 1  # Custom polynomial kernel

    path_res_1 = clf.path(data, pre_computed_affinity, restore_best_weights=False, min_features=data.shape[1] - 3)

    # Now, perform the same path using the integrated kernel
    clf = clf_class(batch_size=50, random_state=0, max_iter=100, kernel="polynomial",
                    kernel_params={"gamma": 1, "coef0": 1, "degree": 1}, alpha=10)
    path_res_2 = clf.path(data, restore_best_weights=False, min_features=data.shape[1] - 3)

    best_weights_1, geminis_1, group_penalties_1, alphas_1, n_features_1 = path_res_1
    best_weights_2, geminis_2, group_penalties_2, alphas_2, n_features_2 = path_res_2

    for a, b in zip(best_weights_1, best_weights_2):
        assert np.all(a == b)

    assert np.all(np.array(geminis_1) == np.array(geminis_2))
    assert np.all(np.array(group_penalties_1) == np.array(group_penalties_2))
    assert np.all(np.array(alphas_1) == np.array(alphas_2))
    assert np.all(np.array(n_features_1) == np.array(n_features_2))


@pytest.mark.parametrize(
    "clf_class",
    [SparseMLPModel, SparseLinearModel]
)
def test_custom_metric_generic(clf_class, data):
    gemini = gemclus.gemini.MMDGEMINI(kernel="precomputed", ovo=False)
    clf = clf_class(batch_size=50, random_state=0, max_iter=100, gemini=gemini, alpha=10)

    pre_computed_affinity = data @ data.T + 1  # Custom polynomial kernel

    path_res_1 = clf.path(data, pre_computed_affinity, restore_best_weights=False, min_features=data.shape[1] - 3)

    # Now, perform the same path using the integrated kernel
    gemini = gemclus.gemini.MMDGEMINI(kernel="polynomial", ovo=False,
                                      kernel_params={"gamma": 1, "coef0": 1, "degree": 1})
    clf = clf_class(batch_size=50, random_state=0, max_iter=100, gemini=gemini, alpha=10)
    path_res_2 = clf.path(data, restore_best_weights=False, min_features=data.shape[1] - 3)

    best_weights_1, geminis_1, group_penalties_1, alphas_1, n_features_1 = path_res_1
    best_weights_2, geminis_2, group_penalties_2, alphas_2, n_features_2 = path_res_2

    for a, b in zip(best_weights_1, best_weights_2):
        assert np.all(a == b)

    assert np.all(np.array(geminis_1) == np.array(geminis_2))
    assert np.all(np.array(group_penalties_1) == np.array(group_penalties_2))
    assert np.all(np.array(alphas_1) == np.array(alphas_2))
    assert np.all(np.array(n_features_1) == np.array(n_features_2))
