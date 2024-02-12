import numpy as np
import pytest
from sklearn import metrics
from sklearn.utils.extmath import softmax

from ..gemini import *
from ..data import celeux_one
from ..gemini._utils import _str_to_gemini

all_geminis = [_str_to_gemini(x) for x in AVAILABLE_GEMINIS if x!="mi"]

@pytest.fixture
def data():
    return celeux_one(n=50, random_state=0)


@pytest.fixture
def fake_data_pred():
    X, y = celeux_one(n=50, random_state=0)
    np.random.seed(0)
    y_logits = np.random.uniform(size=(50, 3))
    return X, y_logits


@pytest.mark.parametrize(
    "objective",
   all_geminis
)
def test_gemini_objective(objective, data):
    X, y = data
    A = objective.compute_affinity(X)
    y_pred = np.eye(len(np.unique(y)))[y]
    loss = objective(y_pred, A)

    assert not np.isnan(loss)
    assert loss > 0
    assert len(loss.shape) == 0

@pytest.mark.parametrize(
    "objective",
    all_geminis
)
def test_empty_clusters_loss(objective, data):
    X, y = data
    A = objective.compute_affinity(X)
    y_pred = np.eye(len(np.unique(y)))[y]
    unperturbed_loss = objective(y_pred, A)

    # Add perturbation (i.e. empty cluster to y)
    y_pred = np.concatenate([y_pred, np.zeros((len(X), 1))], axis=1)
    perturbed_loss = objective(y_pred, A)

    assert np.isclose(unperturbed_loss, perturbed_loss)


@pytest.mark.parametrize(
    "objective",
    all_geminis
)
def test_empty_clusters_gradients(objective, fake_data_pred):
    X, y = fake_data_pred
    A = objective.compute_affinity(X)
    y_pred = softmax(y)
    _, unperturbed_grads = objective(y_pred, A, return_grad=True)

    # Add perturbation (i.e. empty cluster to y)
    y_pred = np.concatenate([y_pred, np.zeros((len(X), 1))], axis=1)
    _, perturbed_grads = objective(y_pred, A, return_grad=True)

    assert np.all(~np.isnan(perturbed_grads))
    assert np.allclose(unperturbed_grads, perturbed_grads[:, :-1])
    assert np.allclose(perturbed_grads[:, -1], 0)


@pytest.mark.parametrize(
    "objective",
    all_geminis
)
def test_existing_gradient(objective, fake_data_pred):
    X, y = fake_data_pred
    A = objective.compute_affinity(X)
    y_pred = softmax(y)
    _, grads = objective(y_pred, A, return_grad=True)

    assert np.max(np.abs(grads)) != 0


@pytest.mark.parametrize(
    "objective, atol",
    [(gemini, 1e-5) for gemini in all_geminis]
)
def test_gradient_precision(objective, atol, fake_data_pred):
    X, y_logits = fake_data_pred

    affinity = objective.compute_affinity(X)

    y_pred = softmax(y_logits)

    loss, pred_grads = objective(y_pred, affinity, return_grad=True)

    logit_grads = y_pred * (pred_grads - (y_pred * pred_grads).sum(1, keepdims=True))

    epsilon = 1e-5

    fake_y_logits = np.copy(y_logits)
    y_grads = np.zeros(y_logits.shape)

    for i in range(y_logits.shape[0]):
        for k in range(y_logits.shape[1]):
            fake_y_logits[i, k] += epsilon

            fake_y_pred = softmax(fake_y_logits)

            new_loss = objective(fake_y_pred, affinity)

            y_grads[i, k] = (new_loss - loss) / epsilon

            fake_y_logits[i, k] -= epsilon

    assert np.allclose(y_grads, logit_grads, atol=atol), f"{y_grads, logit_grads}"

def test_precomputed_kernel(fake_data_pred):
    X, y_logits = fake_data_pred

    kernel_params = {"degree":3, "gamma":2, "coef0":1}

    precomputed_kernel = metrics.pairwise.polynomial_kernel(X, **kernel_params)

    precomputed_objective = MMDGEMINI(kernel="precomputed")
    precomputed_dummy_objective = MMDGEMINI(kernel="precomputed", kernel_params=kernel_params)

    equivalent_objective = MMDGEMINI(kernel="polynomial", kernel_params=kernel_params)

    assert np.all(precomputed_objective.compute_affinity(X, precomputed_kernel)==precomputed_dummy_objective.compute_affinity(X,precomputed_kernel))
    assert np.all(equivalent_objective.compute_affinity(X)==precomputed_objective.compute_affinity(X,precomputed_kernel))
    assert np.all(equivalent_objective.compute_affinity(X, precomputed_kernel)==precomputed_objective.compute_affinity(X,precomputed_kernel))

def test_precomputed_metric(fake_data_pred):
    X, y_logits = fake_data_pred

    precomputed_distance = metrics.pairwise.euclidean_distances(X)

    precomputed_objective = WassersteinGEMINI(metric="precomputed")

    equivalent_objective = WassersteinGEMINI(metric="euclidean")

    assert np.all(equivalent_objective.compute_affinity(X)==precomputed_objective.compute_affinity(X,precomputed_distance))
    assert np.all(equivalent_objective.compute_affinity(X, precomputed_distance)==precomputed_objective.compute_affinity(X,precomputed_distance))

