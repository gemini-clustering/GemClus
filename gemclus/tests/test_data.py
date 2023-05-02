import numpy as np
import pytest

from gemclus.data import *


@pytest.mark.parametrize(
    "method",
    [celeux_one, celeux_two, gstm]
)
def test_random_state(method):
    X1, y1 = method(random_state=0)
    X2, y2 = method(random_state=0)
    print(X1, X2)
    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)


@pytest.mark.parametrize(
    "method, samples",
    [(celeux_one, 1), (celeux_two, 1), (gstm, 4)]
)
def test_few_samples(method, samples):
    X, y = method(n=samples, random_state=0)
    assert len(X) == samples
    assert len(y) == samples


@pytest.mark.parametrize(
    "mean_a",
    [np.zeros(2), [1, 2], [1.0, 2.0],
     np.zeros(2, dtype=int)
     ]
)
@pytest.mark.parametrize(
    "mean_b",
    [np.zeros(2), [1, 2], [1.0, 2.0],
     np.zeros(2, dtype=int)
     ]
)
@pytest.mark.parametrize(
    "n",
    [1, 100, 1000]
)
@pytest.mark.parametrize(
    "pis",
    [np.ones(2) / 2, [0.5, 0.5]]
)
@pytest.mark.parametrize(
    "cov",
    [[np.eye(2), np.eye(2)], np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])]
)
def test_good_params_2d_gmm(n, mean_a, mean_b, cov, pis):
    X, y = draw_gmm(n, [mean_a, mean_b], cov, pis)
    assert X.shape[0] == n
    assert len(X) == len(y)
    assert X.shape[1] == 2


@pytest.mark.parametrize(
    "mean_a",
    [np.zeros(1), [1], [1.0],
     np.zeros(1, dtype=int)
     ]
)
@pytest.mark.parametrize(
    "mean_b",
    [np.zeros(1), [1], [1.0],
     np.zeros(1, dtype=int)
     ]
)
@pytest.mark.parametrize(
    "n",
    [1, 100, 1000]
)
@pytest.mark.parametrize(
    "pis",
    [np.ones(2) / 2, [0.5, 0.5]]
)
@pytest.mark.parametrize(
    "cov",
    [[[1], [1]], np.ones((2, 1))]
)
def test_good_params_1d_gmm(n, mean_a, mean_b, cov, pis):
    X, y = draw_gmm(n, [mean_a, mean_b], cov, pis)
    assert X.shape[0] == n
    assert len(X) == len(y)
    assert X.shape[1] == 1


@pytest.mark.parametrize(
    "mean",
    [np.zeros(2), [1, 2], [1.0, 2.0],
     np.zeros(2, dtype=int)
     ]
)
@pytest.mark.parametrize(
    "n",
    [1, 100, 1000]
)
@pytest.mark.parametrize(
    "df",
    [0.5, 1, 10]
)
@pytest.mark.parametrize(
    "cov",
    [np.eye(2), [[1, 0], [0, 1]], [[1.0, 0.0], [0.0, 1.0]]]
)
def test_good_params_2d_student(n, mean, cov, df):
    print(n, mean, cov, df)
    X = multivariate_student_t(n, mean, cov, df)
    assert X.shape[0] == n
    assert X.shape[1] == 2


@pytest.mark.parametrize(
    "mean, cov",
    [(np.zeros((3, 2)), [np.eye(2)] * 2),  # different number of components
     ([np.zeros(2)] * 2, [np.eye(2), np.zeros((2, 2))]),  # one non-psd covariance
     ]
)
def test_bad_params_gmm(mean, cov):
    try:
        X, y = draw_gmm(n=200, loc=mean, scale=cov, pvals=np.ones(len(mean)) / len(mean))
    except AssertionError as e:
        pass
    except Exception as e:
        print(e)
        assert False
