import itertools
from abc import ABC
from functools import reduce
from numbers import Integral, Real

import numpy as np

from gemclus._constraints import constraint_params
from gemclus.gemini import WassersteinGEMINI, MMDGEMINI, MI, WassersteinOvO
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils import check_array, check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted


class Douglas(ClusterMixin, BaseEstimator, ABC):
    """
    Implementation of the `DNDTs optimised using GEMINI leveraging apprised splits` tree algorithm. This model learns
    clusters by optimising learnable parameters to perform feature-wise soft-binnings and recombine those bins
    into a single cluster predictions. The parameters are optimised to maximise a generalised mutual information score.

    Parameters
    ----------

    n_clusters : int, default=3
        The number of clusters to form as well as the number of output neurons in the neural network.

    n_cuts: int, default=1
        The number of cuts to consider per feature in the soft binning function of the DNDT

    feature_mask: array of boolean [shape d], default None
        A boolean vector indicating whether a feature should be considered or not among splits. If None,
        all features are considered during training.

    temperature: float, default=0.1
        The temperature controls the relative importance of logits per leaf soft-binning. A high temperature smoothens
        the differences in probability whereas a low temperature produces distributions closer to delta Dirac
        distributions.

    max_iter: int, default=100
        The number of epochs for training the model parameters.

    batch_size: int, default=None
        The number of samples per batch during an epoch. If set to `None`, all samples will be considered in a single
        batch.

    learning_rate: float, default=1e-2
        The learning rate hyperparameter for the optimiser's update rule.

    gemini: gemclus.gemini.MMDGEMINI, gemclus.gemini.WassersteinGEMINI or gemclus.gemini.MI instance
        The generalised mutual information objective to maximise w.r.t. the tree parameters. If set to `None`, the
        one-vs-one Wasserstein is chosen.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for feature exploration.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    labels_: ndarray of shape (n_samples,)
        The cluster in which each sample of the data was put
    tree_: Tree instance
        The underlying Tree object. Please refer to `help(sklearn.tree._tree.Tree)` for attributes of Tree object.

    References
    -----------
    FIXME: to be announced
    """
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "n_cuts": [Interval(Integral, 1, None, closed="left"), None],
        "feature_mask": [np.ndarray, None],
        "temperature": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "learning_rate": [Interval(Real, 0, None, closed="neither"), None],
        "gemini": [WassersteinGEMINI, MMDGEMINI, MI, None],
        "verbose": [bool],
        "random_state": ["random_state"]
    }

    def __init__(self, n_clusters=3, n_cuts=1, feature_mask=None, temperature=0.1, max_iter=100, batch_size=None,
                 learning_rate=1e-2, gemini=None, verbose=False, random_state=None):
        self.n_clusters = n_clusters
        self.n_cuts = n_cuts
        self.feature_mask = feature_mask
        self.temperature = temperature
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gemini = gemini
        self.verbose = verbose
        self.random_state = random_state

    def _leaf_binning(self, X, cut_points, temperature=0.1):
        n = len(cut_points)
        W = np.expand_dims(np.linspace(1, n + 1, n + 1, dtype=np.float64), axis=0)
        order = np.argsort(cut_points)
        sorted_cut_points = cut_points[order]
        b = np.cumsum(np.concatenate([np.zeros(1), -sorted_cut_points])).reshape((1, -1))

        logits = X @ W + b

        return softmax(logits / temperature), order

    def _merge_leaf(self, leaf_res1, leaf_res2):
        # Compute feature-wise kronecker product
        product = np.einsum("ij,ik->ijk", leaf_res1, leaf_res2)

        # reshape to 2d
        return product.reshape((-1, np.prod(product.shape[1:])))

    def fit(self, X, y=None):
        """Performs the DOUGLAS algorithm by optimising feature-wise soft-binnings leafs to maximise a chosen GEMINI
        objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        y : ndarray of shape (n_samples, n_samples), default=None
            Use this parameter to give a precomputed affinity metric if the option "precomputed" was passed during
            construction. Otherwise, it is not used and present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        # Check that X has the correct shape
        X = check_array(X)
        X = self._validate_data(X, accept_sparse=True, dtype=np.float64, ensure_min_samples=self.n_clusters)

        # Create the random state
        random_state = check_random_state(self.random_state)

        batch_size = self.batch_size if self.batch_size is not None else len(X)
        batch_size = min(batch_size, len(X))

        # Create the parameters
        if self.feature_mask is None:
            self.cut_points_list_ = [(i, random_state.normal(size=(self.n_cuts,))) for i in range(X.shape[1])]
            num_leaf = int((self.n_cuts + 1) ** X.shape[1])
        else:
            assert len(self.feature_mask) == X.shape[1], ("The boolean feature mask must have as "
                                                          "much entries as the number of features")
            self.cut_points_list_ = [(i, random_state.normal(size=self.n_cuts, )) for i in range(X.shape[1])
                                     if self.feature_mask[i]]
            num_leaf = int((self.n_cuts + 1) ** len(self.cut_points_list_))

        if self.verbose:
            print(self.cut_points_list_)
            print(f"Total will be {num_leaf} values per sample")
        self.leaf_scores_ = random_state.normal(size=(num_leaf, self.n_clusters))

        weights = [self.leaf_scores_] + list(map(lambda x: x[1], self.cut_points_list_))
        self.optimiser_ = AdamOptimizer(weights, self.learning_rate)

        if self.gemini is None:
            gemini = WassersteinOvO(metric="euclidean")
        else:
            gemini = self.gemini

        affinity = gemini.compute_affinity(X, y)

        # Training algorithm
        for epoch in range(self.max_iter):
            batch_idx = 0
            epoch_batch_order = random_state.permutation(len(X))
            while batch_idx * batch_size < len(X):
                batch_indices = epoch_batch_order[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                X_batch = X[batch_indices]

                if affinity is not None:
                    affinity_batch = affinity[batch_indices][:, batch_indices]
                else:
                    affinity_batch = None

                y_pred, leaf, all_binnings, all_orders = self._infer(X_batch, return_intermediates=True)
                # Get probabilities from tree logits
                y_pred = softmax(y_pred)

                # Apply loss function, or rather immediately get gradients
                loss, grads = gemini(y_pred, affinity_batch, return_grad=True)

                # Compute backpropagation

                # Start by the backprop through the softmax
                y_pred_grad = y_pred * (grads - (y_pred * grads).sum(1, keepdims=True))

                # Then backprop through the final matrix multiplication
                binning_backprop = y_pred_grad @ self.leaf_scores_.T
                leaf_score_backprop = leaf.T @ y_pred_grad

                # Store the update corresponding to the leaf score, negate for maximisation instead of minimisation
                updates = [-leaf_score_backprop]

                # Then, compute all feature kronecker derivatives
                axes_for_reshape = tuple([-1] + [len(x[1]) + 1 for x in self.cut_points_list_])
                binning_backprop = binning_backprop.reshape(axes_for_reshape)
                # We must multiply the binning gradient by all binnings
                binning_backprop *= leaf.reshape(axes_for_reshape)

                # Compute individual cut points backprop
                for i, (_, cut_points) in enumerate(self.cut_points_list_):
                    axes_for_sum = tuple([1 + j for j in range(len(self.cut_points_list_)) if i != j])
                    softmax_grad = binning_backprop.sum(axes_for_sum) / all_binnings[i]

                    bin_grad = all_binnings[i] * (
                            softmax_grad - (all_binnings[i] * softmax_grad).sum(1, keepdims=True))  # Shape Nx(d+1)
                    bin_grad /= self.temperature

                    # Gradient is directly on the bias, so we only need to do the cumsum backprop after summing on
                    # all samples

                    # We remove the gradient on the constant [0]
                    bias_grad = bin_grad.sum(0)[1:]
                    cumsum_grad = -np.cumsum(bias_grad[::-1])[::-1]

                    # Take the order back
                    cut_grad = cumsum_grad[np.argsort(all_orders[i])]

                    # Apply update rule and negate for maximisation instead of minimisation
                    updates += [-cut_grad]

                # Update params
                self.optimiser_.update_params(weights, updates)

                batch_idx += 1

            if self.verbose:
                print(f"Epoch: {epoch} / Loss: {loss}")

        batch_idx = 0
        self.labels_ = []
        while batch_idx * batch_size < len(X):
            section = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            X_batch = X[section]
            self.labels_ += [np.argmax(self._infer(X_batch), axis=1)]
            batch_idx += 1
        self.labels_ = np.concatenate(self.labels_)

        self.labels_ = np.argmax(self._infer(X), axis=1)

        return self

    def _infer(self, X, return_intermediates=False):
        leaf_binning = lambda z: self._leaf_binning(X[:, z[0]:z[0] + 1], z[1], self.temperature)
        cut_iterator = map(leaf_binning, self.cut_points_list_)

        all_binnings_results = list(cut_iterator)
        all_binnings = [x[0] for x in all_binnings_results]
        all_orders = [x[1] for x in all_binnings_results]

        leaf = reduce(self._merge_leaf, all_binnings)

        y_pred = leaf @ self.leaf_scores_

        if return_intermediates:
            return y_pred, leaf, all_binnings, all_orders
        else:
            return y_pred

    def fit_predict(self, X, y=None):
        """Performs the DOUGLAS algorithm by optimising feature-wise soft-binnings leafs to maximise a chosen GEMINI
        objective. Returns the predicted cluster memberships of the data samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        y : ndarray of shape (n_samples, n_samples), default=None
            Use this parameter to give a precomputed affinity metric if the option "precomputed" was passed during
            construction. Otherwise, it is not used and present here for API consistency by convention.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the cluster label for each sample.
        """
        return self.fit(X, y).labels_

    def predict(self, X):
        """ Passes the data samples `X` through the tree structure to assign cluster membership.
        This method can be called only once `fit` or `fit_predict` was performed.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the cluster label for each sample.
        """

        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """ Passes the data samples `X` through the tree structure to assign the probability of belonging to each
        cluster.
        This method can be called only once `fit` or `fit_predict` was performed.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_clusters)
            Vector containing on each row the cluster membership probability of its matching sample.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return softmax(self._infer(X))

    def score(self, X, y=None):
        """
        Return the value of the GEMINI evaluated on the given test data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples, n_samples), default=None
            Use this parameter to give a precomputed affinity metric if the option "precomputed" was passed during
            construction. Otherwise, it is not used and present here for API consistency by convention.

        Returns
        -------
        score : float
            GEMINI evaluated on the output of ``self.predict(X)``.
        """
        check_is_fitted(self)

        y_pred = softmax(self._infer(X))
        if self.gemini is None:
            gemini = WassersteinOvO()
        else:
            gemini = self.gemini
        affinity = gemini.compute_affinity(X, y)

        return gemini(y_pred, affinity)


@constraint_params({
    "binnings": ["array-like"]
})
def _merge_rules(binnings):
    """
    Builds a set of rules that encompasses all binnings of a clustering. The methods returns lower and upper bounds
    of rules that summarises how to be part of the binnings.

    Parameters
    ----------
    binnings: ndarray of shape (n_binnings, n_features)
        A 2d array where each entry describes a set of binning that belongs to a cluster.
        For example, the entry [0,2,1] means that a sample is in a cluster if the feature 0 is in bin 0, feature 1
        in bin 2 and feature 2 in bin 1.

    Returns
    -------
    lower: ndarray of shape (n_rules, n_features)
        A set of lower bounds for rules

    upper: ndarray of shape (n_rules, n_features)
        A set of upper bounds for rules.

    """
    seen = np.zeros(len(binnings), dtype=bool)
    all_lowers, all_uppers = [], []

    while seen.sum() != len(binnings):
        best_rule = [0, 0]
        best_count = 0
        # Rules are built by fixing all features except one where we accept variability
        for d in range(binnings.shape[1]):
            sub_binnings = np.delete(binnings, d, axis=1)
            for i in range(len(seen)):
                # We seek the feature on which to allow variability which leverages the most explanation of
                # the binnings
                similar_items, = np.where(np.all(sub_binnings == sub_binnings[i], axis=1))
                newly_explained = (~seen[similar_items]).sum()
                if newly_explained > best_count:
                    best_count = newly_explained
                    best_rule = [i, d]

        # Create the rules
        segmented_binnings = np.delete(binnings, best_rule[1], axis=1)
        rule_base = segmented_binnings[best_rule[0]]

        matching_binnings, = np.where(np.all(segmented_binnings == rule_base, axis=1))
        sorted_variability = sorted(list(set(binnings[matching_binnings, best_rule[1]])))

        seen[matching_binnings] = True

        start = 0
        for i in range(len(sorted_variability) - 1):
            # We may have discontinuous subsets, so we need to create
            # several rules in such case
            if sorted_variability[i] + 1 != sorted_variability[i + 1]:
                lower = np.copy(binnings[best_rule[0]])
                lower[best_rule[1]] = sorted_variability[start]
                upper = np.copy(binnings[best_rule[0]])
                upper[best_rule[1]] = sorted_variability[i]
                start = i + 1
                all_lowers += [np.expand_dims(lower, axis=0)]
                all_uppers += [np.expand_dims(upper, axis=0)]
        lower = np.copy(binnings[best_rule[0]])
        lower[best_rule[1]] = sorted_variability[start]
        upper = np.copy(binnings[best_rule[0]])
        upper[best_rule[1]] = sorted_variability[len(sorted_variability) - 1]
        all_lowers += [np.expand_dims(lower, axis=0)]
        all_uppers += [np.expand_dims(upper, axis=0)]

    return np.concatenate(all_lowers, axis=0), np.concatenate(all_uppers, axis=0)


def _print_rule(lower, upper, cut_points, feature_names=None):
    print("Rule")
    for index in range(len(lower)):
        if feature_names is None:
            rule = f"X[:, {index}]"
        else:
            rule = f"{feature_names[index]}"
        if lower[index] != 0:
            threshold = cut_points[index][lower[index] - 1]
            rule = f"{threshold} ({lower[index] - 1}) < " + rule
        if upper[index] != len(cut_points[index]):
            threshold = cut_points[index][upper[index]]
            rule = rule + f" < {threshold} ({upper[index]})"
        print(f"\t{rule}")


@constraint_params({
    "douglas_tree": [Douglas],
    "feature_names": ["array-like", None],
    "step": [Interval(Real, 0, None, closed="neither")],
    "simplify": ["bool"]
})
def print_douglas_rules(douglas_tree, feature_names=None, step=1, simplify=True):
    """
    Extracts all cut points from a trained `Douglas` instance and generate prints the set of rules to satisfy
    for each clusters.

    The rules are built as the conjunction of the feature binnings in which a sample must be. The threshold values
    correspond to the cut points. In order to compute the cluster assignment per binning combination, the mean values
    of the binnings ar taken. For the outer binnings, an offset `step` is added.

    Parameters
    ----------
    douglas_tree: Douglas
        A Douglas tree instance that was trained.

    feature_names: array of shape (n_features,) or None
        The name to use to describe the features. If set to None, a default print "X[:,i]" is proposed.

    step: float, default=1
        The offset to add to outer binning values when computing the combinations of binnings for cluster assignments

    simplify: bool, default=True
        Whether to display all combinations of binnings per cluster (False) or to minimise at best the number of rules
        per cluster (True).
    """
    check_is_fitted(douglas_tree)

    X = []
    all_binnings = np.array(list(itertools.product(range(douglas_tree.n_cuts + 1), repeat=douglas_tree.n_features_in_)))
    if feature_names is not None:
        assert len(feature_names)==all_binnings.shape[1], "Please provide as much feature names as features"

    sorted_cut_points = [sorted(x[1]) for x in douglas_tree.cut_points_list_]

    for selection in all_binnings:
        sample = []
        for i in range(douglas_tree.n_features_in_):
            if selection[i] == 0:
                sample += [sorted_cut_points[i][selection[i]] - step]
            elif selection[i] == douglas_tree.n_cuts:
                sample += [sorted_cut_points[i][selection[i] - 1] + step]
            else:
                upper = sorted_cut_points[i][selection[i]]
                lower = sorted_cut_points[i][selection[i] - 1]
                sample += [(upper - lower) / 2]
        X += [np.array(sample)]
    y_pred_rules = douglas_tree.predict(np.array(X))

    for k in np.unique(y_pred_rules):
        print(f"Cluster {k}")
        all_samples, = np.where(y_pred_rules == k)

        if not simplify:
            for i in all_samples:
                _print_rule(all_binnings[i], all_binnings[i], sorted_cut_points, feature_names)
        else:
            lower, upper = _merge_rules(all_binnings[all_samples])
            for lo, up in zip(lower, upper):
                _print_rule(lo, up, sorted_cut_points, feature_names)
