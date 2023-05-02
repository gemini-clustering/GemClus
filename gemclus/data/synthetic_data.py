from numbers import Integral, Real
from typing import Tuple

import numpy as np
from scipy.linalg import block_diag
from sklearn.utils import check_random_state, check_array
from sklearn.utils._param_validation import validate_params, Interval, Iterable


@validate_params(
    {
        "n": [Interval(Integral, 1, None, closed="left")],
        "loc": ["array-like"],
        "scale": ["array-like"],
        "pvals": ["array-like", Iterable[float]],
        "random_state": ["random_state"]
    }
)
def draw_gmm(n, loc, scale, pvals, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns :math:`n` samples drawn from a mixture of Gaussian distributions. The number of components
    is determined by the number of elements in the lists of the parameters.

    Parameters
    ----------
    n: int
        The number of samples to draw from the GMM.
    loc: list of 1d ndarray
        A list containing the means of all components of the Gaussian mixture distributions.
    scale: list of 2d ndarray
        A list containing the covariances of all components of the Gaussian mixture distribution.
    pvals: 1d ndarray
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
    loc = check_array(loc, ensure_2d=True, ensure_min_samples=2, input_name="Means")
    scale = check_array(scale, allow_nd=True, ensure_min_samples=2, input_name="Covariances")
    pvals = check_array(pvals, ensure_2d=False, ensure_min_samples=2, input_name="Proportions")

    # Check that the parameters have satisfying properties
    K, d = loc.shape
    assert K == scale.shape[0], "The means and the covariances do not contain the same number of components"
    assert d == scale.shape[1] and d == scale.shape[1], "The covariances should be square matrices"
    assert K == pvals.shape[0], "The proportions and the means do not contain the same number of components"
    assert np.all(pvals > 0), "Proportions or components should be positive."
    assert np.sum(pvals) == 1, "Proportions of components do not add up to one."

    generator = check_random_state(random_state)

    # Draw samples from each distribution

    X = []

    # Draw the true cluster from which to draw
    y = generator.choice(K, p=pvals, size=(n,))
    if d == 1:
        for k in range(len(loc)):
            X += [generator.normal(loc[k], scale[k], size=(n,))]
    else:
        for k in range(len(loc)):
            X += [generator.multivariate_normal(loc[k], scale[k], size=(n,))]

    X = [X[k][i].reshape((1, -1)) for i, k in enumerate(y)]

    return np.concatenate(X, axis=0), y


@validate_params(
    {
        "n": [Interval(Integral, 1, None, closed="left")],
        "loc": ["array-like"],
        "scale": ["array-like"],
        "df": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"]
    }
)
def multivariate_student_t(n, loc, scale, df=10, random_state=None) -> np.ndarray:
    """
    Draws :math:`n` samples from a multivariate Student-t distribution.

    Parameters
    ----------
    n: int
        The number of samples to draw from the distribution.

    loc: ndarray of shape (d,)
        The position of the distribution to sample from.

    scale: ndarray of shape (d,d)
        Positive semi-definite scale matrix.

    df: int, default=10
        Degrees of freedom of the distribution. Controls the spread of the samples.

    random_state: int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int for reproducible output across
        multiple runs.

    Returns
    -------
    X: ndarray of shape (n,d)
        The samples drawn from the Student-t distribution.
    """
    loc = check_array(loc, ensure_2d=False, input_name="Location")
    scale = check_array(scale, ensure_2d=True, input_name="Scale")

    d = len(loc)
    assert scale.shape[0] == d and scale.shape[1] == d, "Please provide a mean and scale with consistent shapes"

    generator = check_random_state(random_state)

    # Start the sampling process by generating from a 0-mean multivariate distribution
    nx = generator.multivariate_normal(np.zeros(d), scale, size=n)
    u = generator.chisquare(df, n).reshape((-1, 1))
    X = np.sqrt(df / u) * nx + loc.reshape((1, -1))

    return X


@validate_params(
    {
        "n": [Interval(Integral, 4, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="neither")],
        "df": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"]
    }
)
def gstm(n=500, alpha=2, df=1, random_state=None):
    """
    Reproduces the Gaussian-Student Mixture dataset from the GEMINI article.

    Parameters
    ----------
    n: int, default=500
        The number of samples to draw from the dataset.

    alpha: float, default=2:
        This parameter controls how close the means of the Gaussian distribution and the location of the Student-t
        distribution are.

    df: float, default=1
        The degrees of freedom for the Student-t distribution.

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
    GEMINI - Ohl, L., Mattei, P. A., Bouveyron, C., Harchaoui, W., Leclercq, M., Droit, A., & Precioso, F.
        (2022, October). Generalised Mutual Information for Discriminative Clustering. In Advances in Neural
        Information Processing Systems.
    """
    generator = check_random_state(random_state)

    # Build the location and scale of each distribution
    locations = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * alpha
    covariance = np.eye(2)

    # For the 3 Gaussian distribution, we draw the samples using a GMM with proportions 1/3 on 3/4 of the samples
    n_gaussian = 3 * n // 4
    X_gaussian, y_gaussian = draw_gmm(n_gaussian, locations[:-1], [covariance] * 3, np.ones(3) / 3, generator)

    # Then we sample the student-t distribution
    n_student = n - n_gaussian
    X_student = multivariate_student_t(n_student, locations[-1], covariance, df, generator)

    X = np.vstack([X_gaussian, X_student])
    y = np.concatenate([y_gaussian, np.ones(n_student) * 3])

    # Apply one final random permutation to shuffle the data
    order = generator.permutation(n)

    return X[order], y[order]


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
