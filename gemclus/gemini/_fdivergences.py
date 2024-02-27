from abc import ABC
from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval

from .._constraints import constraint_params
from ._base_loss import _GEMINI


class _FDivergence(_GEMINI, ABC):
    # This helper intermediate class simply defines the compute_affinity for all f-divergences
    def compute_affinity(self, X, y=None):
        """
        Unused for f-divergences.

        Returns
        -------
        None
        """
        return None


class KLGEMINI(_FDivergence):
    """
    Implements the one-vs-all and one-vs-one KL GEMINI.

    The one-vs-all version compares the KL divergence between a cluster distribution
    and the data distribution. It is the classical mutual information.

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{KL}(p(x|y)\|p(x))]

    The one-vs-one version compares the KL divergence between two cluster distributions.

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\\text{KL}(p(x|y_a)\|p(x|y_b))]

    Parameters
    ----------
    ovo: bool, default=False
        Whether to use the one-vs-all objective (False) or the one-vs-one objective (True).

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """

    @constraint_params(
        {
            "ovo": [bool],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, ovo=False, epsilon=1e-12):
        super().__init__(epsilon)
        self.ovo = ovo

    def evaluate(self, y_pred, affinity, return_grad=False):
        # Use a clip mask for numerical stability in gradients
        clip_mask = (y_pred > self.epsilon) & (y_pred < 1 - self.epsilon)
        p_y_x = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        p_y = p_y_x.mean(0)

        log_p_y_x = np.log(p_y_x)
        log_p_y = np.log(p_y)

        cluster_entropy = np.sum(p_y * log_p_y)
        prediction_entropy = np.sum(np.mean(p_y_x * log_p_y_x, axis=0))

        if self.ovo:
            mutual_information = prediction_entropy - np.sum(p_y * np.mean(log_p_y_x, axis=0))
        else:
            mutual_information = prediction_entropy - cluster_entropy

        if return_grad:
            if self.ovo:
                gradient_mi = (log_p_y_x + 1) / log_p_y_x.shape[0] - (p_y / p_y_x + np.mean(log_p_y_x, axis=0)) / \
                              log_p_y_x.shape[0]
            else:
                gradient_mi = log_p_y_x / log_p_y_x.shape[0] - log_p_y / log_p_y_x.shape[0]
            return mutual_information, gradient_mi * clip_mask
        else:
            return mutual_information


class MI(KLGEMINI):
    """
    Implements the classical mutual information between cluster conditional probabilities and the complete data
    probabilities:

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{KL}(p(x|y)\|p(x))]

    This class is a simplified shortcut for KLGEMINI(ovo=False).

    Parameters
    ----------
    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """

    def __init__(self, epsilon=1e-12):
        super().__init__(ovo=False, epsilon=epsilon)


class TVGEMINI(_FDivergence):
    """
    Implements the one-vs-all and one-vs-one Total Variation distance GEMINI.

    The one-vs-all version compares the total variation distance between a cluster distribution
    and the data distribution.

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{TV}(p(x|y)\|p(x))]

    The one-vs-one version compares the TV distance between two cluster distributions.

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\\text{TV}(p(x|y_a)\|p(x|y_b))]

    Parameters
    ----------
    ovo: bool, default=False
        Whether to use the one-vs-all objective (False) or the one-vs-one objective (True).

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """

    @constraint_params(
        {
            "ovo": [bool],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, ovo=False, epsilon=1e-12):
        super().__init__(epsilon)
        self.ovo = ovo

    def evaluate(self, y_pred, affinity, return_grad=False):
        # Use a clip mask for numerical stability in gradients
        clip_mask = (y_pred > self.epsilon) & (y_pred < 1 - self.epsilon)
        p_y_x = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        p_y = p_y_x.mean(0)

        if self.ovo:
            # Extend to 3d tensors to compute p(y=k|x_i) * p(y=k^\prime)
            # Shape: NxKx1
            extended_p_y = np.expand_dims(np.repeat(np.expand_dims(p_y, axis=0), y_pred.shape[0], axis=0), axis=-1)
            # Nx1xK
            extended_p_y_x = np.expand_dims(p_y_x, axis=1)
            # NxKxK
            cross_product = extended_p_y @ extended_p_y_x
            difference = cross_product - np.transpose(cross_product, axes=[0, 2, 1])
        else:
            difference = p_y_x - p_y
        sign_mask = np.sign(difference)

        # Doing the mean on the data axis, helps to handle the sum of differences with either 2d or 3d axes
        pseudo_estimates = np.mean(np.abs(difference), axis=0)

        tv_gemini = 0.5 * np.sum(pseudo_estimates)

        if return_grad:
            if self.ovo:
                base_grad = sign_mask / y_pred.shape[0]
                cross_prod_grad = base_grad - np.transpose(base_grad, axes=[0, 2, 1])  # NxKxK
                extended_p_y_x_grad = np.transpose(extended_p_y,
                                                   axes=[0, 2, 1]) @ cross_prod_grad  # Nx1xK,NxKxK => Nx1xK
                extended_p_y_grad = cross_prod_grad @ np.transpose(extended_p_y_x,
                                                                   axes=[0, 2, 1])  # NxKxK, NxKx1 => NxKx1

                gradients = np.squeeze(extended_p_y_x_grad) + np.squeeze(extended_p_y_grad).mean(0)
            else:
                gradients = (sign_mask - np.mean(sign_mask, axis=0)) / y_pred.shape[0]
            return tv_gemini, 0.5 * gradients * clip_mask
        else:
            return tv_gemini


class HellingerGEMINI(_FDivergence):
    """
    Implements the one-vs-all and one-vs-one Squared Hellinger distance GEMINI.

    The one-vs-all version compares the squared Hellinger distance distance between a cluster distribution
    and the data distribution.

    .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{H}^2(p(x|y)\|p(x))]

    The one-vs-one version compares the squared Hellinger distance between two cluster distributions.

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\\text{H}^2(p(x|y_a)\|p(x|y_b))]

    Parameters
    ----------
    ovo: bool, default=False
        Whether to use the one-vs-all objective (False) or the one-vs-one objective (True).

    epsilon: float, default=1e-12
        The precision for clipping the prediction values in order to avoid numerical instabilities.
    """

    @constraint_params(
        {
            "ovo": [bool],
            "epsilon": [Interval(Real, 0, 1, closed="neither")]
        }
    )
    def __init__(self, ovo=False, epsilon=1e-12):
        super().__init__(epsilon)
        self.ovo = ovo

    def evaluate(self, y_pred, affinity, return_grad=False):
        # Use a clip mask for numerical stability in gradients
        clip_mask = (y_pred > self.epsilon) & (y_pred < 1 - self.epsilon)
        p_y_x = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        p_y = p_y_x.mean(0)

        cluster_wise_estimates = np.sqrt(p_y_x * p_y)
        estimates = np.sum(cluster_wise_estimates, axis=1)

        if self.ovo:
            estimates = np.square(estimates)

        hellinger_gemini = 1 - np.mean(estimates, axis=0)

        if return_grad:
            if self.ovo:
                sqrt_estimates = np.sqrt(estimates.reshape((-1, 1)))
                gradients = - (p_y / cluster_wise_estimates * sqrt_estimates + np.mean(
                    p_y_x / cluster_wise_estimates * sqrt_estimates, axis=0))
            else:
                gradients = -0.5 * (p_y / cluster_wise_estimates + np.mean(p_y_x / cluster_wise_estimates, axis=0))

            gradients /= y_pred.shape[0]
            return hellinger_gemini, gradients * clip_mask
        else:
            return hellinger_gemini
