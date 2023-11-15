import pytest
import sklearn.exceptions

import gemclus
from sklearn.utils.estimator_checks import check_estimator, check_clustering

import numpy as np
from gemclus.gemini import MMDOvA, MMDOvO, WassersteinOvA, WassersteinOvO

from sklearn import metrics

from gemclus.tree import Kauri, Douglas, print_kauri_tree


@pytest.mark.parametrize(
    ["X", "y"], [gemclus.data.celeux_one(n=300, p=1, random_state=0)]
)
@pytest.mark.timeout(5)
def test_running_kauri(X, y):
    print(X)

    model = Kauri(max_clusters=3, verbose=True)

    y_pred = model.fit_predict(X)

    print(metrics.adjusted_rand_score(y_pred, y))


@pytest.mark.parametrize(
    ["X", "y"], [gemclus.data.celeux_one(n=300, p=5, random_state=0)]
)
def test_many_variables_kauri(X, y):
    print(X)

    model = Kauri(max_clusters=3, verbose=True)

    y_pred = model.fit_predict(X)

    print(metrics.adjusted_rand_score(y_pred, y))

    print("Used variables: ")
    import numpy as np
    print(np.unique([x for x in model.tree_.features if x is not None]))


@pytest.mark.parametrize(
    ["X", "y"], [gemclus.data.gstm(n=300, random_state=0)]
)
def test_min_leaf_kauri(X, y):
    print(X)

    model = Kauri(max_clusters=4, min_samples_leaf=10, min_samples_split=20, verbose=True)

    y_pred = model.fit_predict(X)

    print(metrics.adjusted_rand_score(y_pred, y))

    print(y_pred)


@pytest.mark.parametrize(
    ["X", "y"], [gemclus.data.gstm(n=300, random_state=0)]
)
def test_max_depth_kauri(X, y):
    print(X)

    model = Kauri(max_clusters=4, min_samples_split=20, max_depth=2, verbose=True)

    y_pred = model.fit_predict(X)

    print(metrics.adjusted_rand_score(y_pred, y))

    print(y_pred)

    print(model.score(X))


def test_kauri_estimator():
    check_estimator(Kauri())


def test_douglas_estimator():
    estimator = Douglas()
    check_iterator = check_estimator(estimator, generate_only=True)
    for clf, check in check_iterator:
        if check.func == check_clustering:
            continue
        check(clf)


@pytest.mark.parametrize(
    "gemini", [MMDOvA(), MMDOvO(), WassersteinOvA(), WassersteinOvO()]
)
@pytest.mark.timeout(5)
def test_geminis_douglas(gemini):
    X, y = gemclus.data.celeux_one(n=300, p=1, random_state=0)

    clf = Douglas(verbose=True, random_state=0, gemini=gemini)
    y_pred = clf.fit_predict(X)

    print(metrics.adjusted_rand_score(y_pred, y))
    print(clf.find_active_points(X))


def test_batches_douglas():
    X, y = gemclus.data.celeux_one(n=1000, p=1, random_state=0)

    clf = Douglas(verbose=True, random_state=0, gemini=MMDOvA(), batch_size=100, max_iter=10)
    y_pred = clf.fit_predict(X)

    print(metrics.adjusted_rand_score(y_pred, y))


def test_kauri_printing():
    model = Kauri()

    with pytest.raises(sklearn.exceptions.NotFittedError) as excinfo:
        print_kauri_tree(model)

    X, y = gemclus.data.celeux_one(n=5, random_state=0)
    model.fit(X)

    print_kauri_tree(model)

    feature_names = list(range(X.shape[1]))
    print_kauri_tree(model, feature_names)
    print_kauri_tree(model, np.array(feature_names))
