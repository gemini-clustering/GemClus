import numpy as np


def soft_threshold(l, x):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)


def mlp_prox_grad(W_skip_, W1_, alpha, M):
    v = W_skip_
    u = W1_

    # Contrary to the original version, here the output dimension is on the columns
    # Not the rows
    u_abs_sorted = np.sort(np.abs(u), axis=1)[:, ::-1]  # shape dxh

    batch, k = u.shape

    s = np.arange(k + 1.0).reshape((1, -1))  # Shape 1x(h+1)
    zeros = np.zeros((batch, 1))  # Shape dx1

    a_s = alpha - M * np.concatenate(
        [zeros, np.cumsum(u_abs_sorted, axis=1)],
        axis=1
    )  # Shape dx(h+1)

    norm_v = np.linalg.norm(v, ord=2, axis=1, keepdims=True)  # shape dx1

    x = np.maximum(1 - a_s / norm_v, 0) / (1 + s * M ** 2)  # Shape dx(h+1)

    w = M * x * norm_v
    intervals = soft_threshold(0, u_abs_sorted)  # Shape dxh
    lower = np.concatenate([intervals, zeros], axis=1)  # Shape dx(h+1)

    idx = np.sum(lower > w, axis=1, keepdims=True)  # Shape dx1

    x_star = np.take_along_axis(x, idx, axis=1).reshape((batch, 1))  # Shape dx1
    w_star = np.take_along_axis(w, idx, axis=1).reshape((batch, 1))

    beta_star = x_star * v
    theta_star = np.where(u >= 0, 1, -1) * np.minimum(soft_threshold(0, np.abs(u)), w_star)

    return beta_star, theta_star


def group_mlp_prox_grad(groups, W_skip, W1, alpha, M):
    W_skip_star = np.empty(W_skip.shape)
    W1_star = np.empty(W1.shape)

    for g in groups:
        group_W_skip = W_skip[g]
        group_W1 = W1[g]
        group_W_skip_star, group_W1_star = mlp_prox_grad(
            group_W_skip.reshape((1, -1)),
            group_W1.reshape((1, -1)),
            alpha,
            M
        )
        W_skip_star[g] = group_W_skip_star.reshape(group_W_skip.shape)
        W1_star[g] = group_W1_star.reshape(group_W1.shape)

    return W_skip_star, W1_star


def linear_prox_grad(W, alpha):
    # Shape of W is [d,h]
    # alpha is a scalar
    W_norms = np.linalg.norm(W, axis=1, keepdims=True)  # Shape [d,1]

    # Group lasso soft thresholding
    W_star = np.maximum(W_norms - alpha, 0) * W / np.where(W_norms == 0, 1, W_norms)

    return W_star


def group_linear_prox_grad(groups, W, alpha):
    W_star = np.empty(W.shape)

    for g in groups:
        group_W = W[g]
        group_W_star = linear_prox_grad(group_W.reshape((1, -1)), alpha)
        W_star[g] = group_W_star.reshape(group_W.shape)

    return W_star
