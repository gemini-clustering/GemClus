from numbers import Integral, Real

import numpy as np
from scipy.linalg import block_diag
from typing import Tuple

from sklearn.utils import check_random_state
from sklearn.utils._param_validation import validate_params, Interval, Iterable


@validate_params(
    {
        "n": [Interval(Integral, 1, None, closed="left")],
        "mus": ["array-like"],
        "sigmas": ["array-like"],
        "pis": ["array-like", Iterable[float]],
        "random_state": ["random_state"]
    }
)
def draw_gmm(n, mus, sigmas, pis, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns :math:`n` samples drawn from a mixture of Gaussian distributions. The number of components
    is determined by the number of elements in the lists of the parameters.

    Parameters
    ----------
    n: int
        The number of samples to draw from the GMM.
    mus: list of 1d ndarray
        A list containing the means of all components of the Gaussian mixture distributions.
    sigmas: list of 2d ndarray
        A list containing the covariances of all components of the Gaussian mixture distribution.
    pis: 1d ndarray
        The proportions of each component of the Gaussian mixture.
    random_state: int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int for reproducible output across
        multiple runs.

    Returns
    -------
    X: ndarray
        The array containing the samples drawn from the mixture model.
    y: ndarray
        The component from which each sample originates.
    """

    # Check that the parameters have satisfying properties
    assert len(mus) == len(sigmas), "The means and the covariances do not contain the same number of components"
    assert len(pis) == len(sigmas), "The proportions and the covariances do not contain the same number of components"
    assert len(pis) > 1, "The gmm requires at least two components."
    main_dim = len(mus[0])
    for k in range(len(pis)):
        assert len(mus[k]) == len(sigmas[k]), f"Inconsistent shape between of {k}-th means and covariances"
        assert len(mus[k]) == main_dim, f"Inconsistent dimensions between all components. Should be {main_dim} or" \
                                        f" {len(mus[k])} everywhere."
        assert pis[k] > 0, f"Proportions or components should be positive."
        # Other checks regarding psd for covariances are included in the multivariate_normal function

    assert sum(pis) == 1, "Proportions of components do not add up to one."

    generator = check_random_state(random_state)

    # Draw samples from each distribution

    X = []
    if main_dim == 1:
        X += [generator.normal(mus[k], sigmas[k], size=(n,))]
    else:
        for k in range(len(mus)):
            X += [generator.multivariate_normal(mus[k], sigmas[k], size=(n,))]

    # Draw the true cluster from which to draw
    y = np.random.choice(len(mus), p=pis, size=(n,))

    X = [X[k][i].reshape((1, -1)) for i, k in enumerate(y)]

    return np.concatenate(X, axis=0), y


@validate_params(
    {
        "n": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Integral, 1, None, closed="left")],
        "mu": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"]
    }
)
def celeux_one(n=300, p=20, mu=1.7, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws :math:`n` samples from a Gaussian mixture with 3 isotropic components of respective means 1, 0 and 1
    over 5 dimensions scaled by :math:`\mu`. The data is concatenated with :math:`p` additional noisy excessive random
    variables that are independent of the true clusters. This dataset is taken by Celeux et al., section 3.1.

    Parameters
    ----------
    n: int, default=300
        The number of samples to draw from the gaussian mixture models.
    p: int, default=20
        The number of excessive noisy variables to concatenate to the dataset.
    mu: float, default=1.7
        Controls how the means of the components are close to each other by scaling.
    random_state: int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int for reproducible output across
        multiple runs.

    Returns
    -------
    X: ndarray
        The samples of the dataset in an array of shape n_samples x n_features
    y: ndarray
        The component of the GMM from which each sample was drawn.

    References
    -----------
    Dataset - Celeux, G., Martin-Magniette, M. L., Maugis-Rabusseau, C., & Raftery, A. E. (2014). Comparing model
        selection and regularization approaches to variable selection in model-based clustering.
        Journal de la Societe francaise de statistique, 155(2), 57-71.
    """
    generator = check_random_state(random_state)

    # Draw the first five variables according to a balance gaussian mixture
    mu1 = np.ones(5) * mu
    mu2 = -mu1
    mu3 = np.zeros(5)
    cov = np.eye(5)

    good_variables, y = draw_gmm(n, [mu1, mu2, mu3], [cov, cov, cov], np.ones(3) / 3, generator)

    # Create noisy independent variables
    noise = generator.normal(size=(n, p))

    return np.concatenate([good_variables, noise], axis=1), y


@validate_params(
    {
        "n": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"]
    }
)
def celeux_two(n=2000, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws samples from a mixture of 4 Gaussian distributions in 2d with additional variables
    linearly dependent of the informative variables and non-informative noisy variables. This dataset is
    taken from Celeux et al., section 3.2.

    Parameters
    ----------
    n: int, default=2000
        The number of samples to draw.
    random_state: int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int for reproducible output across
        multiple runs.

    Returns
    -------
    X: ndarray
        The samples of the dataset in an array of shape n_samples x n_features
    y: ndarray
        The component of the GMM from which each sample was drawn.

    References
    -----------
    Dataset - Celeux, G., Martin-Magniette, M. L., Maugis-Rabusseau, C., & Raftery, A. E. (2014). Comparing model
        selection and regularization approaches to variable selection in model-based clustering.
        Journal de la Societe francaise de statistique, 155(2), 57-71.
    """

    generator = check_random_state(random_state)

    # Start by generating the true informative variables
    mu1 = np.array([0, 0])
    mu2 = np.array([4, 0])
    mu3 = np.array([0, 2])
    mu4 = np.array([4, 2])
    cov = np.eye(2)
    pis = np.ones(4) / 4
    good_variables, y = draw_gmm(n, [mu1, mu2, mu3, mu4], [cov, cov, cov, cov], pis, generator)

    # Apply affine transformations to produce correlated variables up to some noise
    b = np.array([[0.5, 1], [2, 0], [0, 3], [-1, 2], [2, -4], [0.5, 0], [4, 0.5], [3, 0], [2, 1]]).T
    rot_pi_3 = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    rot_pi_6 = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    cov_noise = [np.eye(3), 0.5 * np.eye(2)]
    cov_noise += [rot_pi_3.T @ np.diag(np.array([1, 3])) @ rot_pi_3]
    cov_noise += [rot_pi_6.T @ np.diag(np.array([2, 6])) @ rot_pi_6]
    cov_noise = block_diag(*cov_noise)
    noise = generator.multivariate_normal(np.zeros(9), cov_noise, size=(n,))
    X3_11 = np.array([0, 0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]) + good_variables @ b + noise

    # Add noisy indepedent variables
    X12_14 = generator.multivariate_normal(np.array([3.2, 3.6, 4]), np.eye(3), size=(n,))

    # Complete the dataset by joining everything
    bad_variables = np.concatenate([X3_11, X12_14], axis=1)
    return np.concatenate([good_variables, bad_variables], axis=1), y
