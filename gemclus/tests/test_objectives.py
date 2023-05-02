import numpy as np
import pytest
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.utils.extmath import softmax

from .._gemini_grads import wasserstein_ova, wasserstein_ovo, mmd_ova, mmd_ovo
from ..data import celeux_one

wasserstein_objectives = [wasserstein_ova, wasserstein_ovo]
mmd_objectives = [mmd_ova, mmd_ovo]


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
    "objective, affinity_fct",
    [(mmd_ova, pairwise_kernels), (mmd_ovo, pairwise_kernels),
     (wasserstein_ova, pairwise_distances), (wasserstein_ovo, pairwise_distances)]
)
def test_gemini_objective(objective, affinity_fct, data):
    X, y = data
    A = affinity_fct(X)
    y_pred = np.eye(len(np.unique(y)))[y]
    loss = objective(y_pred, A)

    assert not np.isnan(loss)
    assert loss > 0


@pytest.mark.parametrize(
    "objective, affinity_fct",
    [(mmd_ova, pairwise_kernels), (mmd_ovo, pairwise_kernels),
     (wasserstein_ova, pairwise_distances), (wasserstein_ovo, pairwise_distances)]
)
def test_empty_clusters_loss(objective, affinity_fct, data):
    X, y = data
    A = affinity_fct(X)
    y_pred = np.eye(len(np.unique(y)))[y]
    unperturbed_loss = objective(y_pred, A)

    # Add perturbation (i.e. empty cluster to y)
    y_pred = np.concatenate([y_pred, np.zeros((len(X), 1))], axis=1)
    perturbed_loss = objective(y_pred, A)

    assert np.isclose(unperturbed_loss, perturbed_loss)


@pytest.mark.parametrize(
    "objective, affinity_fct",
    [(mmd_ova, pairwise_kernels), (mmd_ovo, pairwise_kernels),
     (wasserstein_ova, pairwise_distances), (wasserstein_ovo, pairwise_distances)]
)
def test_empty_clusters_gradients(objective, affinity_fct, fake_data_pred):
    X, y = fake_data_pred
    A = affinity_fct(X)
    y_pred = softmax(y)
    _, unperturbed_grads = objective(y_pred, A, return_grad=True)

    # Add perturbation (i.e. empty cluster to y)
    y_pred = np.concatenate([y_pred, np.zeros((len(X), 1))], axis=1)
    _, perturbed_grads = objective(y_pred, A, return_grad=True)

    assert np.all(~np.isnan(perturbed_grads))
    assert np.allclose(unperturbed_grads, perturbed_grads[:, :-1])
    assert np.allclose(perturbed_grads[:, -1], 0)


@pytest.mark.parametrize(
    "objective, affinity_fct",
    [(mmd_ova, pairwise_kernels), (mmd_ovo, pairwise_kernels),
     (wasserstein_ova, pairwise_distances), (wasserstein_ovo, pairwise_distances)]
)
def test_existing_gradient(objective, affinity_fct, fake_data_pred):
    X, y = fake_data_pred
    A = affinity_fct(X)
    y_pred = softmax(y)
    _, grads = objective(y_pred, A, return_grad=True)

    assert np.max(np.abs(grads)) != 0


@pytest.mark.parametrize(
    "objective, affinity_fct, atol",
    [(mmd_ova, pairwise_kernels, 1e-5), (mmd_ovo, pairwise_kernels, 1e-5),
     (wasserstein_ova, pairwise_distances, 1e-5), (wasserstein_ovo, pairwise_distances, 1e-5)]
)
def test_gradient_precision(objective, affinity_fct, atol, fake_data_pred):
    X, y_logits = fake_data_pred

    affinity = affinity_fct(X)

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
