from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval

from .._constraints import constraint_params
from ._base_loss import _GEMINI


class KLGEMINI(_GEMINI):
    """
    Implements the one-vs-all and one-vs-one KL GEMINI.

    The one-vs-all version compares the KL divergence between a cluster distribution
    and the data distribution. It is the classical mutual information.

        .. math::
        \mathcal{I} = \mathbb{E}_{y \sim p(y)}[\\text{KL}(p(x|y)\|p(x))]

    The one-vs-one version compares the KL divergence distance between two cluster distributions.

    .. math::
        \mathcal{I} = \mathbb{E}_{y_a,y_b \sim p(y)}[\text{KL}(p(x|y_a)\|p(x|y_b))]

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
            mutual_information = prediction_entropy - np.sum(p_y*np.mean(log_p_y_x, axis=0))
        else:
            mutual_information = prediction_entropy - cluster_entropy

        if return_grad:
            if self.ovo:
                gradient_mi = (log_p_y_x + 1)/log_p_y_x.shape[0] - (p_y/p_y_x + np.mean(log_p_y_x, axis=0)) / log_p_y_x.shape[0]
            else:
                gradient_mi = log_p_y_x / log_p_y_x.shape[0] - log_p_y / log_p_y_x.shape[0]
            return mutual_information, gradient_mi*clip_mask
        else:
            return mutual_information

    def compute_affinity(self, X, y=None):
        """
        Unused for f-divergences.

        Returns
        -------
        None
        """
        return None


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
