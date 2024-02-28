import warnings

import numpy as np
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer
from sklearn.utils import check_random_state


def check_groups(groups, n_features_in):
    if groups is not None:
        all_indices = []
        for g in groups:
            all_indices.extend(list(g))
        # Ensure that the indices are valid
        if min(all_indices) < 0 or max(all_indices) >= n_features_in:
            raise ValueError(f"Indices passed to the groups argument should be contained in [0, {n_features_in}]")
        if len(all_indices) == n_features_in:
            # We expect that it is a partition, so we should have as much indices as the number of features
            if set(all_indices) != set(range(n_features_in)):
                raise ValueError("Groups must form a partition of the set of variable indices. Perhaps there are duplicate indices?")
            return groups
        else:
            # Not all indices are covered by the proposal, so let's assure that they are unique, then complete
            if len(set(all_indices)) != len(all_indices):
                raise ValueError("There cannot be duplicate entries in groups.")
            new_groups = groups + [[i] for i in range(n_features_in) if i not in all_indices]
            return new_groups
    else:
        return None


def compute_val_score(clf, X, y, batch_size, gemini_objective):
    validation_gemini = 0
    validation_l1 = clf._group_lasso_penalty() * clf.alpha
    selection_mask = np.arange(X.shape[1])
    if clf.dynamic and y is None:
        selection_mask = clf.get_selection()
    j = 0
    while j < len(X):
        X_batch = X[j:j + batch_size]
        if y is not None:
            affinity = y[j:j+batch_size][:,j:j+batch_size]
        else:
            affinity = gemini_objective.compute_affinity(X_batch[:, selection_mask])
        y_pred = clf.predict_proba(X_batch)
        validation_gemini += gemini_objective(y_pred, affinity) * len(X_batch)
        j += batch_size
    validation_gemini /= len(X)
    return validation_gemini, validation_l1


def _path(clf, X, y=None, alpha_multiplier=1.05, min_features=2, keep_threshold=0.9,
          early_stopping_factor=0.99, max_patience=10):
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
    clf.fit(X, y)

    generator = check_random_state(clf.random_state)
    if clf.batch_size is not None:
        batch_size = clf.batch_size
    else:
        batch_size = len(X)

    gemini_objective = clf.get_gemini()
    best_gemini_score, _ = compute_val_score(clf, X, y, batch_size, gemini_objective)  # Best gemini score only when using all features
    weights = clf._get_weights()
    best_weights = [w.copy() for w in weights]

    if clf.verbose:
        print(f"Finished initial training. GEMINI = {best_gemini_score}")

    affinity = gemini_objective.compute_affinity(X, y)

    alphas = []
    n_features = []
    geminis = []
    group_lasso_penalties = []

    # Re-initialise the optimiser to SGD with 0.9 momentum (default option), to follow the torch version
    # nesterov acceleration is set to True by default
    clf.optimiser_ = SGDOptimizer(weights, clf.learning_rate)

    while clf._n_selected_features() > min_features:
        clf.alpha = alpha

        # Compute the validation scores at the beginning of this step of the path
        validation_gemini_score, validation_l1 = compute_val_score(clf, X, y, batch_size, gemini_objective)

        if clf.verbose:
            print(f"Starting new iteration with: alpha = {clf.alpha}. Validation score is {validation_gemini_score}")

        if clf.dynamic and y is None:
            selection_mask = clf.get_selection()
            partial_data = X[:, selection_mask]
            affinity = gemini_objective.compute_affinity(partial_data)

        patience = 0
        i = 0
        while i < clf.max_iter and patience < max_patience:

            for X_batch, affinity_batch in clf._batchify(X, affinity, generator):
                y_pred = clf._infer(X_batch)
                _, grads = gemini_objective(y_pred, affinity_batch, return_grad=True)
                grads = clf._compute_grads(X_batch, y_pred, grads)
                clf._update_weights(weights, grads)

            # Epoch control
            iteration_gemini_score, iteration_l1 = compute_val_score(clf, X, y, batch_size, gemini_objective)

            if iteration_gemini_score > (2 - early_stopping_factor) * validation_gemini_score \
                    or iteration_l1 < early_stopping_factor * validation_l1:
                validation_l1 = iteration_l1
                validation_gemini_score = iteration_gemini_score
                patience = 0
            else:
                patience += 1
            if np.isnan(iteration_gemini_score):
                warnings.warn(f"Unfortunately, the GEMINI converged to nan, making the entire path unsucessful."
                              f"Please report this error. Score and gradients are: {iteration_gemini_score}, {grads}")
                patience = max_patience

            i += 1

        if np.isnan(iteration_gemini_score):
            break

        alphas.append(alpha)
        n_features.append(clf._n_selected_features().item())
        geminis.append(iteration_gemini_score)
        group_lasso_penalties.append(clf._group_lasso_penalty())

        if clf.verbose:
            print(f"Finished after {i} iterations. Current iteration score is {iteration_l1 - iteration_gemini_score}. "
                  f"\t(GEMINI: {iteration_gemini_score}, L1: {iteration_l1}). Number of features is"
                  f" {clf._n_selected_features().item()}")

        alpha *= alpha_multiplier
        if iteration_gemini_score >= best_gemini_score and clf._n_selected_features() == X.shape[1]:
            best_gemini_score = iteration_gemini_score
            if clf.verbose:
                print("Best GEMINI score so far using all features, saving it.")

        if iteration_gemini_score >= keep_threshold * best_gemini_score:
            best_weights = [w.copy() for w in weights]
            if clf.verbose:
                print(f"This is definitely the best score so far within threshold: {iteration_gemini_score}, "
                      f"{best_gemini_score}")

    return best_weights, geminis, group_lasso_penalties, alphas, n_features
