import warnings
from abc import ABC
from numbers import Real

import numpy as np
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import softmax

from .._gemini_grads import linear_prox_grad, group_mlp_prox_grad, group_linear_prox_grad, mlp_prox_grad
from ..linear._linear_geminis import _LinearGEMINI
from ..mlp._mlp_geminis import _MLPGEMINI


def check_groups(groups, n_features_in):
    if groups is not None:
        all_indices = []
        for g in groups:
            all_indices.extend(list(g))
        assert len(all_indices) == n_features_in and set(all_indices) == set(range(n_features_in)), \
            f"Groups must form a partition of the set of variable indices"

def compute_val_score(clf,X, batch_size):
    validation_gemini = 0  # clf.score(X)
    validation_l1 = clf._group_lasso_penalty() * clf.alpha
    j = 0
    while j < len(X):
        X_batch = X[j:j + batch_size]
        validation_gemini += clf.score(X_batch) * len(X_batch)
        j += batch_size
    validation_gemini /= len(X)
    return validation_gemini, validation_l1

def _path(clf, X, alpha_multiplier=1.05, min_features=2, keep_threshold=0.9,
          early_stopping_factor=0.99, max_patience=10):
    clf._validate_data(X)
    check_groups(clf.groups, X.shape[1])
    if alpha_multiplier <= 1:
        warnings.warn(f"The alpha multiplier is lower or equal to 1. This will not increase alpha during the path. "
                      f"Setting it again to default parameters: 1.05")
        alpha_multiplier = 1.05
    if keep_threshold < 0 or keep_threshold > 1:
        warnings.warn(f"The threshold to keep the best solution is outside [0,1]: {keep_threshold}, setting it "
                      f"to default: 0.9")
        keep_threshold = 0.9
    if min_features <= 0:
        warnings.warn(f"The min_features to stop the path iterations is below 0 which implies infinite loop. "
                      f"Setting it to default: 2")
        min_features = 2
    elif min_features >= X.shape[1]:
        warnings.warn(f"The min_features param is greater or equal to the number of features. This implies that "
                      f"no path will be performed. The method is equivalent to `fit`.")

    # Start by fitting the model using all features and without regularisation
    alpha = clf.alpha
    clf.set_params(alpha=0)

    if clf.verbose:
        print("Starting initial training with alpha = 0")
    clf.fit(X)
    best_gemini = clf.score(X)
    weights = clf._get_weights()
    best_weights = [w.copy() for w in weights]

    if clf.verbose:
        print(f"Finished initial training. GEMINI = {best_gemini}")

    kernel = clf._compute_affinity(X)

    generator = check_random_state(clf.random_state)
    if clf.batch_size is not None:
        batch_size = clf.batch_size
    else:
        batch_size = len(X)

    alphas = [0]
    n_features = [X.shape[1]]
    geminis = [best_gemini]
    group_lasso_penalties = [clf._group_lasso_penalty()]

    # Re-initialise the optimiser to SGD with 0.9 momentum (default option)
    clf.optimiser_ = SGDOptimizer(weights, clf.learning_rate)

    while clf._n_selected_features() > min_features:
        clf.alpha = alpha

        # Compute the validation scores at the beginning of this step of the path
        validation_gemini, validation_l1 = compute_val_score(clf, X, batch_size)

        if clf.verbose:
            print(f"Starting new iteration with: alpha = {clf.alpha}. Validation score is {validation_gemini}")

        gemini_score = np.array([validation_gemini])
        patience = 0
        i = 0
        while i < clf.max_iter and patience < max_patience:
            all_indices = generator.permutation(len(X))
            j = 0
            while j < len(X):
                batch_indices = all_indices[j:j + batch_size]
                X_batch = X[batch_indices]
                kernel_batch = kernel[batch_indices][:, batch_indices]
                y_pred = clf._infer(X_batch)
                gemini_score, grads = clf._compute_gemini(y_pred, kernel_batch, return_grad=True)
                grads = clf._compute_grads(X_batch, y_pred, grads)
                clf._update_weights(weights, grads)

                j += batch_size

            iteration_gemini, iteration_l1 = compute_val_score(clf, X, batch_size)

            if iteration_gemini > (2 - early_stopping_factor) * validation_gemini \
                    or iteration_l1 < early_stopping_factor * validation_l1:
                validation_l1 = iteration_l1
                validation_gemini = iteration_gemini
                patience = 0
            else:
                patience += 1
            if np.isnan(gemini_score):
                warnings.warn(f"Unfortunately, the GEMINI converged to nan, making the entire path unsucessful."
                              f"Please report this error. Score and gradients are: {gemini_score}, {grads}")
                patience = max_patience

            i += 1

        alphas.append(alpha)
        n_features.append(clf._n_selected_features().item())
        geminis.append(gemini_score.item())
        group_lasso_penalties.append(clf._group_lasso_penalty())

        if clf.verbose:
            print(f"Finished after {i} iterations. Current iteration score is {iteration_gemini - iteration_l1}. "
                  f"Current GEMINI score is {gemini_score}. Number of features is"
                  f" {clf._n_selected_features().item()}")

        alpha *= alpha_multiplier
        if gemini_score >= best_gemini:
            best_gemini = gemini_score
            if clf.verbose:
                print("Best GEMINI score so far, saving it.")

        if gemini_score >= keep_threshold * best_gemini:
            best_weights = [w.copy() for w in weights]
            if clf.verbose:
                print(f"This is definitely the best score so far within threshold: {gemini_score}, {best_gemini}")

    return best_weights, geminis, group_lasso_penalties, alphas, n_features


class _SparseMLPGEMINI(_MLPGEMINI, ABC):
    """ This is the BaseSparseGEMINI template to derive to create a Sparse GEMINI MLP clustering model.
    When deriving, the only methods to adapt is the _compute_gemini methods which
    should be able to return the gradient with respect to the conditional distribution p(y|x).

    On top of the vanilla MLP GEMINI model, this variation brings a skip connection from the data to the cluster
    output. This skip connection ensures a sparsity constraint through a group-lasso penalty and a proximal gradient
    that eliminates input features as well in the first layer of the MLP.

    This architecture is inspired from LassoNet by Lemhadri et al. (2021).

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    groups: list of arrays of various shapes, default=None
        If groups is set, it must describe a partition of the indices of variables. This will be used for performing
        variable selection with groups of features considered to represent one variables. This option can typically be
        used for one-hot-encoded variables.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    n_hidden_dim: int, default=20
        The number of neurons in the hidden layer of the neural network.

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    alpha: float, default=1e-2
        The weight of the group-lasso penalty in the optimisation scheme.

    M: float, default=10 The hierarchy coefficient that controls the relative strength between the group-lasso
        penalty of the skip connection and the sparsity of the first layer of the MLP.

    batch_size: int, default=None
        The size of batches during gradient descent training. If set to None, the whole data will be considered.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for weights and bias initialisation.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    W1_: ndarray, shape (n_features, n_hidden_dim)
        The linear weights of the first layer
    b1_: ndarray of shape (1, n_hidden_dim)
        The biases of the first layer
    W2_: ndarray of shape (n_hidden_dim, n_clusters)
        The linear weights of the hidden layer
    b2_: ndarray of shape (1, n_clusters)
        The biases of the hidden layer
    W_skip_: ndarray of shape (n_features, n_clusters)
        The linear weights of the skip connection
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    H_: ndarray of shape (n_samples, n_hidden_dim)
        The hidden representation of the samples after fitting.
    """
    _parameter_constraints: dict = {
        **_MLPGEMINI._parameter_constraints,
        "M": [Interval(Real, 0, np.inf, closed="left")],
        "lambda_": [Interval(Real, 0, np.inf, closed="neither")],
    }

    def __init__(self, n_clusters=3, groups=None, max_iter=1000, learning_rate=1e-3, n_hidden_dim=20, M=10,
                 alpha=1e-2, solver="adam", batch_size=None, verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_hidden_dim=n_hidden_dim,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.M = M
        self.alpha = alpha
        self.groups = groups

    def _init_params(self, random_state):
        super()._init_params(random_state)
        threshold = np.sqrt(1 / self.n_features_in_)
        self.W_skip_ = random_state.uniform(-threshold, threshold, size=(self.n_features_in_, self.n_clusters))

    def _infer(self, X, retain=True):
        H = np.maximum(X @ self.W1_ + self.b1_, 0)
        output_network = H @ self.W2_ + self.b2_
        output_skip = X @ self.W_skip_

        if retain:
            self.H_ = H

        return softmax(output_network + output_skip)

    def _compute_grads(self, X, y_pred, gradient):

        tau_hat_grad = y_pred * (gradient - (y_pred * gradient).sum(1, keepdims=True))  # Shape NxK

        W2_grad = self.H_.T @ tau_hat_grad  # Shape
        b2_grad = tau_hat_grad.sum(0, keepdims=True)

        backprop_grad = tau_hat_grad @ W2_grad.T
        backprop_grad *= self.H_ > 0
        W1_grad = X.T @ backprop_grad
        b1_grad = backprop_grad.sum(0, keepdims=True)

        W_skip_grad = X.T @ tau_hat_grad  # Gradient from the GEMINI objective

        # Negative sign to force the optimiser to maximise instead of minimise
        gradients = [-W1_grad, -W2_grad, -W_skip_grad, -b1_grad, -b2_grad]

        return gradients

    def _get_weights(self):
        return [self.W1_, self.W2_, self.W_skip_, self.b1_, self.b2_]

    def _update_weights(self, weights, gradients):
        # First update the weights according to our optimiser
        self.optimiser_.update_params(weights, gradients)

        # Then statisfy the sparsity constraint of the MLP by
        # evaluating the proximal gradient
        if self.groups is None:
            new_W_skip, new_W1 = mlp_prox_grad(self.W_skip_, self.W1_, self.alpha * self.optimiser_.learning_rate,
                                               self.M)
        else:
            new_W_skip, new_W1 = group_mlp_prox_grad(self.groups, self.W_skip_, self.W1_,
                                                     self.alpha * self.optimiser_.learning_rate, self.M)

        np.copyto(self.W_skip_, new_W_skip)
        np.copyto(self.W1_, new_W1)

    def _n_selected_features(self):
        return (np.linalg.norm(self.W_skip_, axis=1, ord=2) != 0).sum()

    def get_selection(self):
        """
        Retrieves the indices of features that were selected by the model.

        Returns
        -------
        ind: ndarray
            The indices of the selected features.
        """
        return np.nonzero(np.linalg.norm(self.W_skip_, axis=1, ord=2))[0]

    def _group_lasso_penalty(self):
        return np.linalg.norm(self.W_skip_, axis=1, ord=2).sum()

    def path(self, X, alpha_multiplier=1.05, min_features=2, keep_threshold=0.9, restore_best_weights=True,
             early_stopping_factor=0.99, max_patience=10):
        """
        Unfold the progressive geometric increase of the penalty weight starting from the initial alpha until
        there remains only a specified amount of features.

        The history of the different gemclus scores are kept as well as the best weights with minimum of features
        ensuring that the GEMINI score remains at a certain percentage of the maximum GEMINI score seen during the
        path.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples on which the feature reduction will be made.
        alpha_multiplier : float, default=1.05
            The geometric increase of the group-lasso penalty at each-retraining. It must be greater than 1.
        min_features: int, default=2
            The number of features that must remain at best to stop performing the path.
        keep_threshold: float, default=0.9
            The percentage of the maximal GEMINI under which any solution with a minimal number of features is deemed
            best.
        restore_best_weights: bool, default=True
            After performing the path, the best weights offering simultaneously good GEMINI score and few features
            are restored to the model.
        early_stopping_factor: float, default=0.99
            The percentage factor beyond which upgrades of the GEMINI or the group-lasso penalty are considered
            too small for early stopping.
        max_patience:
            The maximum number of iterations to wait without any improvements in either the gemclus score or the
            group-lasso penalty before stopping the current step.

        Returns
        -------
        best_weights: list of ndarray of various shapes of length 5
            The list containing the best weights during the path. Sequentially: `W1_`, `W2_`, `W_skip_`, `b1_`, `b2_`
        geminis: list of float of length T
            The history of the gemclus scores as the penalty alpha was increased.
        group_penalties: list of float of length T
            The history of the group-lasso penalties
        alphas: list of float of length T
            The history of the penalty alphas during the path.
        n_features: list of float of length T
            The number of features that were selected at step t.
        """
        best_weights, geminis, group_lasso_penalties, alphas, n_features = _path(self, X, alpha_multiplier,
                                                                                 min_features, keep_threshold,
                                                                                 early_stopping_factor, max_patience)

        if restore_best_weights:
            if self.verbose:
                print("Restoring best weights")
            np.copyto(self.W1_, best_weights[0])
            np.copyto(self.W2_, best_weights[1])
            np.copyto(self.W_skip_, best_weights[2])
            np.copyto(self.b1_, best_weights[3])
            np.copyto(self.b2_, best_weights[4])

        return best_weights, geminis, group_lasso_penalties, alphas, n_features


class _SparseLinearGEMINI(_LinearGEMINI, ABC):
    """ This is the SparseLinearGEMINI clustering model.
    When deriving, the only methods to adapt is the _compute_gemini methods which
    should be able to return the gradient with respect to the conditional distribution p(y|x).

    On top of the vanilla Linear GEMINI model, this variation brings a group-lasso penalty constraint to ensure
    feature selection via a proximal gradient during training.

    Parameters
    ----------
    n_clusters : int, default=3
        The maximum number of clusters to form as well as the number of output neurons in the neural network.

    groups: list of arrays of various shapes, default=None
        If groups is set, it must describe a partition of the indices of variables. This will be used for performing
        variable selection with groups of features considered to represent one variables. This option can typically be
        used for one-hot-encoded variables.

    max_iter: int, default=1000
        Maximum number of epochs to perform gradient descent in a single run.

    learning_rate: float, default=1e-3
        Initial learning rate used. It controls the step-size in updating the weights.

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    alpha: float, default=1e-2
        The weight of the group-lasso penalty in the optimisation scheme.

    batch_size: int, default=None
        The size of batches during gradient descent training. If set to None, the whole data will be considered.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for weights and bias initialisation.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    W_: ndarray of shape (n_features, n_clusters)
        The linear weights of model
    b_: ndarray of shape (1, n_clusters)
        The biases of the model
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    """
    _parameter_constraints: dict = {
        **_LinearGEMINI._parameter_constraints,
        "M": [Interval(Real, 0, np.inf, closed="left")],
        "lambda_": [Interval(Real, 0, np.inf, closed="neither")],
    }

    def __init__(self, n_clusters=3, groups=None, max_iter=1000, learning_rate=1e-3, alpha=1e-2, batch_size=None,
                 solver="adam", verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.alpha = alpha
        self.groups = groups

    def _update_weights(self, weights, gradients):
        # First update the weights according to our optimiser
        self.optimiser_.update_params(weights, gradients)

        # Then statisfy the sparsity constraint of the MLP by
        # evaluating the proximal gradient
        if self.groups is None:
            new_W = linear_prox_grad(self.W_, self.alpha * self.optimiser_.learning_rate)
        else:
            new_W = group_linear_prox_grad(self.groups, self.W_, self.alpha * self.optimiser_.learning_rate)

        np.copyto(self.W_, new_W)

    def _n_selected_features(self):
        return (np.linalg.norm(self.W_, axis=1, ord=2) != 0).sum()

    def get_selection(self):
        """
        Retrieves the indices of features that were selected by the model.

        Returns
        -------
        ind: ndarray
            The indices of the selected features.
        """
        return np.nonzero(np.linalg.norm(self.W_, axis=1, ord=2))

    def _group_lasso_penalty(self):
        return np.linalg.norm(self.W_, axis=1, ord=2).sum()

    def path(self, X, alpha_multiplier=1.05, min_features=2, keep_threshold=0.9, restore_best_weights=True,
             early_stopping_factor=0.99, max_patience=10):
        """
        Unfold the progressive geometric increase of the penalty weight starting from the initial alpha until
        there remains only a specified amount of features.

        The history of the different gemclus scores are kept as well as the best weights with minimum of features
        ensuring that the GEMINI score remains at a certain percentage of the maximum GEMINI score seen during the
        path.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples on which the feature reduction will be made.
        alpha_multiplier : float, default=1.05
            The geometric increase of the group-lasso penalty at each-retraining. It must be greater than 1.
        min_features: int, default=2
            The number of features that must remain at best to stop performing the path.
        keep_threshold: float, default=0.9
            The percentage of the maximal GEMINI under which any solution with a minimal number of features is deemed
            best.
        restore_best_weights: bool, default=True
            After performing the path, the best weights offering simultaneously good GEMINI score and few features
            are restored to the model.
        early_stopping_factor: float, default=0.99
            The percentage factor beyond which upgrades of the GEMINI or the group-lasso penalty are considered
            too small for early stopping.
        max_patience:
            The maximum number of iterations to wait without any improvements in either the gemclus score or the
            group-lasso penalty before stopping the current step.

        Returns
        -------
        best_weights: list of ndarray of various shapes of length 5
            The list containing the best weights during the path. Sequentially: `W_`, `b_`
        geminis: list of float of length T
            The history of the gemclus scores as the penalty alpha was increased.
        group_penalties: list of float of length T
            The history of the group-lasso penalties
        alphas: list of float of length T
            The history of the penalty alphas during the path.
        n_features: list of float of length T
            The number of features that were selected at step t.
        """
        best_weights, geminis, group_lasso_penalties, alphas, n_features = _path(self, X, alpha_multiplier,
                                                                                 min_features, keep_threshold,
                                                                                 early_stopping_factor, max_patience)

        if restore_best_weights:
            if self.verbose:
                print("Restoring best weights")
            np.copyto(self.W_, best_weights[0])
            np.copyto(self.b_, best_weights[1])

        return best_weights, geminis, group_lasso_penalties, alphas, n_features
