import numpy as np
import ot
from sklearn.utils._param_validation import validate_params


@validate_params(
    {
        "y_pred": ["array-like"],
        "K": ["array-like"],
        "return_grad": [bool]
    }
)
def mmd_ova(y_pred, K, return_grad=False):
    # To stabilise the computations and in order to avoid the
    # softmax overconfident values which would lead to division by 0
    # we, clamp the predictions
    epsilon = 1e-12
    clip_mask = (y_pred > epsilon) & (y_pred < (1 - epsilon))
    y_pred = np.clip(y_pred, a_min=epsilon, a_max=1 - epsilon)

    N = y_pred.shape[0]
    normalised_kernel = K / N ** 2

    pi = y_pred.mean(0, keepdims=True)

    alpha = y_pred / pi
    gamma = normalised_kernel @ alpha
    omega = alpha * gamma

    a = omega.sum(0)
    b = gamma.sum(0)
    c = normalised_kernel.sum()

    delta = np.sqrt(np.maximum(a + c - 2 * b, 0))  # For numerical stability, keep if positive

    mmd_ova_value = np.dot(pi, delta)

    if return_grad:
        tau_grad = (np.eye(N) - 1 / N) @ normalised_kernel @ (alpha - 1)
        delta_mask = (delta == 0)
        gradient = tau_grad / (delta + delta_mask).reshape((1, -1))
        gradient[:, delta_mask] = 0
        return mmd_ova_value, gradient * clip_mask
    else:
        return mmd_ova_value


@validate_params(
    {
        "y_pred": ["array-like"],
        "K": ["array-like"],
        "return_grad": [bool]
    }
)
def mmd_ovo(y_pred, K, return_grad=False):
    # To stabilise the computations and in order to avoid the
    # softmax overconfident values which would lead to division by 0
    # we, clamp the predictions
    epsilon = 1e-12
    clip_mask = (y_pred > epsilon) & (y_pred < 1 - epsilon)
    y_pred = np.clip(y_pred, a_min=epsilon, a_max=1 - epsilon)

    N = y_pred.shape[0]
    normalised_kernel = K / N ** 2

    pi = y_pred.mean(0, keepdims=True)

    alpha = y_pred / pi
    gamma = normalised_kernel @ alpha
    omega = alpha.T @ gamma

    A = np.diag(omega).reshape((1, -1))

    delta = np.sqrt(np.maximum(-2 * omega + A + A.T, 0))  # For numerical stability, keep if positive

    mmd_ovo_value = pi @ delta @ pi.T

    if return_grad:
        # Some distances may be equal to 0 (including self-distances)
        # So we need to remove them from the gradient
        Lambda = (pi.T @ pi) / (delta + np.eye(len(delta)))
        Lambda -= np.diag(np.diag(Lambda))
        Lambda[delta == 0] = 0

        gradient = gamma * Lambda.sum(0, keepdims=True)
        gradient -= gamma @ Lambda
        gradient -= A * Lambda.sum(0, keepdims=True) / N
        gradient += (alpha * (gamma @ Lambda)).mean(0)
        gradient /= pi
        gradient += pi @ delta / N
        gradient *= 2

        return mmd_ovo_value, gradient * clip_mask
    else:
        return mmd_ovo_value


@validate_params(
    {
        "y_pred": ["array-like"],
        "D": ["array-like"],
        "return_grad": [bool]
    }
)
def wasserstein_ova(y_pred, D, return_grad=False):
    N, K = y_pred.shape

    # To stabilise the computations and in order to avoid the
    # softmax overconfident values which would lead to division by 0
    # we, clamp the predictions
    epsilon = 1e-12
    clip_mask = (y_pred > epsilon) & (y_pred < 1 - epsilon)
    y_pred = np.clip(y_pred, a_min=epsilon, a_max=1 - epsilon)

    pi = y_pred.mean(0)

    wy = np.ascontiguousarray((y_pred / (pi.reshape((1, -1)) * N)).T)

    constant_weights = np.ones(N) / N

    wasserstein_distances = np.zeros(y_pred.shape[1])
    dual_variables = [None] * y_pred.shape[1]

    for k in range(K):
        wasserstein_distances[k], dual_variables[k] = ot.emd2(wy[k], constant_weights, D, log=True)

    wasserstein_ova_value = np.dot(pi, wasserstein_distances)
    if return_grad:
        u_bar = np.vstack([x["u"] - x["u"].mean() for x in dual_variables]).T
        grads = u_bar / N + wasserstein_distances / N
        grads -= (y_pred * u_bar).sum(0) / (N * N * pi)
        return wasserstein_ova_value, grads * clip_mask
    else:
        return wasserstein_ova_value


@validate_params(
    {
        "y_pred": ["array-like"],
        "D": ["array-like"],
        "return_grad": [bool]
    }
)
def wasserstein_ovo(y_pred, D, return_grad=False):
    N, K = y_pred.shape

    if return_grad:
        grads = np.zeros(y_pred.shape)

    # To stabilise the computations and in order to avoid the
    # softmax overconfident values which would lead to division by 0
    # we, clamp the predictions
    epsilon = 1e-12
    clip_mask = (y_pred > epsilon) & (y_pred < 1 - epsilon)
    y_pred = np.clip(y_pred, a_min=epsilon, a_max=1 - epsilon)

    pi = y_pred.mean(0)

    wy = np.ascontiguousarray((y_pred / (pi.reshape((1, -1)) * N)).T)

    wasserstein_distances = np.zeros((K, K))

    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            emd, log = ot.emd2(wy[k1], wy[k2], D, log=True)
            wasserstein_distances[k1, k2] = emd
            wasserstein_distances[k2, k1] = emd

            if return_grad:
                u_bar = log["u"] - log["u"].mean()
                v_bar = log["v"] - log["v"].mean()
                grads[:, k1] += 2 * pi[k2] * (u_bar / N - (u_bar * y_pred[:, k1] / (N * N * pi[k1])).sum())
                grads[:, k2] += 2 * pi[k1] * (v_bar / N - (v_bar * y_pred[:, k2] / (N * N * pi[k2])).sum())

    wasserstein_ovo_value = np.dot(pi, np.dot(wasserstein_distances, pi))

    if return_grad:
        grads += 2 * np.dot(wasserstein_distances, pi) / N
        return wasserstein_ovo_value, grads * clip_mask
    else:
        return wasserstein_ovo_value
