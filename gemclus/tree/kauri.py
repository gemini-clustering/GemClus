import warnings

import numpy as np

from abc import ABC
from numbers import Integral

from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.utils import check_array, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ._utils import find_best_split, gemini_objective, Split
from .._constraints import constraint_params


class Tree:
    def __init__(self):
        self.children_left = [-1]
        self.children_right = [-1]
        self.target = [0]
        self.thresholds = [None]
        self.features = [None]
        self.gains = [0]
        self.depths = [0]
        self.n_nodes = 1
        self.categorical_nodes = [False]

    def _add_child(self, father: int, split: Split):  # feature, threshold, gain, target, left_to_target):
        # Update the values of the father
        self.children_left[father] = self.n_nodes
        self.children_right[father] = self.n_nodes + 1
        self.thresholds[father] = split.threshold
        self.features[father] = split.feature
        self.gains[father] = split.gain
        self.categorical_nodes[father] = split.is_categorical

        # Extend the lists to incorporate two children
        self.children_left += [-1, -1]
        self.children_right += [-1, -1]
        self.thresholds += [None, None]
        self.features += [None, None]
        self.gains += [0, 0]
        self.depths += [self.depths[father] + 1, self.depths[father] + 1]
        self.categorical_nodes += [False, False]

        self.target += [split.left_target, split.right_target]

        self.n_nodes += 2

    def get_depth(self, node=None):
        if node is None:
            return max(self.depths)
        else:
            node = min(max(node, 0), len(self.depths))
            return self.depths[node]

    def __len__(self):
        return self.n_nodes

    def predict(self, X, node=0):
        assert 0 <= node <= self.n_nodes, f"Cannot explore tree from unexisting node {node}"
        if self.children_left[node] == -1:
            return self.target[node] * np.ones(len(X), dtype=np.int64)
        else:
            if self.categorical_nodes[node]:
                X_left = X[:, self.features[node]] == self.thresholds
            else:
                X_left = X[:, self.features[node]] <= self.thresholds[node]
            X_right = ~X_left

            predictions = np.zeros(len(X), dtype=np.int64)
            predictions[X_left] = self.predict(X[X_left], self.children_left[node])
            predictions[X_right] = self.predict(X[X_right], self.children_right[node])

            return predictions


class Kauri(ClusterMixin, BaseEstimator, ABC):
    """
    Implementation of the `KMeans as unsupervised reward ideal` tree algorithm. This model learns clusters by
    iteratively performing splits on different nodes of the tree and either assigning those nodes to new clusters
    or refurbishing them to already existing one according to some kernel-guided gain scores.

    Parameters
    ----------

    max_clusters : int, default=3
        The maximum number of clusters to form.

    max_depth: int, default=None
        The maximum depth to limit the tree construction. If set to `None`, then the tree is not limited in depth.

    min_samples_split: int, default=2
        The minimum number of samples that must be contained in a leaf node to consider splitting it into two new
        leaves.

    min_samples_leaf: int, default=1
        The minimum number of samples that must be at least in a leaf. Note that the logical constraint
        `min_samples_leaf`*2 <= `min_samples_split` must be satisfied.

    max_features: int, default=None
        The maximal number of features (randomly selected) to consider upon the choice of splitting a leaf.
        If set to `None`, then all features of the data will be used.

    max_leaves: int, default=None
        The maximal number of leaves that can be found in the tree. If set to `None`, then the tree is not limited
        in number of leaves.

    kernel: {'additive_chi2', 'chi2', 'cosine','linear','poly','polynomial','rbf','laplacian','sigmoid', 'precomputed'},
        default='linear'
        The kernel to use in combination with the MMD objective. It corresponds to one value of `KERNEL_PARAMS`.
        Currently, all kernel parameters are the default ones.
        If set to 'precomputed', then a custom kernel must be passed to the `y` argument of the `fit` or `fit_predict`
        method.


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
    ----------
    KAURI - End-to-end training of unsupervised trees
        Louis Ohl, Pierre-Alexandre Mattei, Mickaël Leclercq, Arnaud Droit, Frederic Preciosio
    """
    _parameter_constraints: dict = {
        "max_clusters": [Interval(Integral, 1, None, closed="left")],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "min_samples_split": [Interval(Integral, 2, None, closed="left")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "max_features": [Interval(Integral, 1, None, closed="left"), None],
        "max_leaves": [Interval(Integral, 2, None, closed="left"), None],
        "kernel": [StrOptions(set(list(PAIRWISE_KERNEL_FUNCTIONS) + ["precomputed"])), callable],
        "verbose": [bool],
        "random_state": ["random_state"]
    }

    def __init__(self, max_clusters=3, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None,
                 max_leaves=None, kernel="linear", verbose=False, random_state=None):
        self.max_clusters = max_clusters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaves = max_leaves
        self.kernel = kernel
        self.verbose = verbose
        self.random_state = random_state

    def _compute_kernel(self, X, y=None):
        if self.kernel == "precomputed":
            if y is None:
                warnings.warn("A precomputed kernel was supposed to be passed to arg y, yet y is None... "
                              "Switching to linear kernel")
                kernel = pairwise_kernels(X, metric="linear")
            else:
                kernel = y
        else:
            kernel = pairwise_kernels(X, metric=self.kernel)
        return kernel

    def fit(self, X, y=None):
        """Performs the KAURI algorithm by repeatedly choosing leaves, evaluating best gain and increasing the tree
        structure until structural limits or maximal gains are reached.

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
        X = self._validate_data(X, accept_sparse=True, dtype=np.float64, ensure_min_samples=self.min_samples_leaf)

        # Create the random state
        random_state = check_random_state(self.random_state)

        # Check that all variables follow some logical constraints
        assert self.min_samples_leaf * 2 <= self.min_samples_split, (f"Contradiction between the number of samples "
                                                                     f"required to consider a split and the number of"
                                                                     f" samples needed to create a leaf")

        if self.verbose:
            print("Initialising variables")

        kernel = self._compute_kernel(X, y)

        n, self.n_features_in_ = X.shape

        # Set up variables for tree construction
        max_leaves = self.max_leaves if self.max_leaves is not None else n
        max_features = min(X.shape[1], max(self.max_features, 1)) if self.max_features is not None else X.shape[1]
        max_depth = len(X) if self.max_depth is None else self.max_depth

        # Set up variables for tree representation
        self.tree_ = Tree()

        Z = np.zeros((max_leaves, len(X)), dtype=np.int64)  # Leaf2sample
        Z[0, :] = 1
        Y = np.zeros((self.max_clusters, max_leaves), dtype=np.int64)  # Cluster 2 leaf
        Y[0, 0] = 1

        # Trackers to help minimising computations
        n_leaves = 1
        n_clusters = 1

        leaves_to_explore = [0]
        last_gain = np.inf

        leaf2node = {0: 0}

        if self.verbose:
            print("Starting main loop")

        while last_gain > 0 and n_leaves < max_leaves and len(leaves_to_explore) != 0:
            # This is going to be inside the loop

            best_split = find_best_split(kernel,
                                         X,
                                         np.array(leaves_to_explore),
                                         Y,
                                         Z,
                                         n_clusters,
                                         self.max_clusters,
                                         n_leaves,
                                         self.min_samples_leaf,
                                         random_state.choice(X.shape[1], size=max_features, replace=False))

            last_gain = best_split.gain

            if last_gain > 0:
                if self.verbose:
                    print(f"Gain is: {last_gain}")
                    print(f"=> Cut is on feature {best_split.feature} <= {best_split.threshold}")
                    print(f"=> From ({best_split.leaf}), assignments are L = {best_split.left_target}"
                          f" / R = {best_split.right_target}")
                # Update our knowledge given the split

                # Find the indices of this leaf
                leaf_indices, = np.where(Z[best_split.leaf] == 1)

                # Now, identify indices of left vs right split
                left_indices, = np.where(X[leaf_indices, best_split.feature] <= best_split.threshold)
                left_indices = leaf_indices[left_indices]
                right_indices = np.setxor1d(leaf_indices, left_indices)

                if self.verbose:
                    print(f"=> Sizes are: L = {len(left_indices)} / R = {len(right_indices)}")

                # Start by updating Z
                # Left always keep the same leaf number, i.e. turn of right in this leaf
                Z[best_split.leaf, right_indices] = 0
                # Right gets added a new leaf number
                Z[n_leaves, right_indices] = 1

                # Then Y
                # Find the custer of leaf
                k = Y[:, best_split.leaf].argmax()
                Y[k, best_split.leaf] = 0  # Leaf does not belong any longer in this cluster
                Y[best_split.left_target, best_split.leaf] = 1  # It belongs to the target of left
                Y[best_split.right_target, n_leaves] = 1

                # Update the tree using the split
                self.tree_._add_child(leaf2node[best_split.leaf], best_split)
                parent_depth = self.tree_.get_depth(leaf2node[best_split.leaf])

                # Update the leaf 2 node

                # # At each split, we add 2 nodes. So we will always have 2*n_leaves-1 nodes (e.g. 2 leaves => 3
                # nodes, 4 leaves => 7 nodes)
                leaf2node[best_split.leaf] = 2 * n_leaves - 1  # Index of the left child
                leaf2node[n_leaves] = 2 * n_leaves  # Index of the right child

                # Pop out the old leaf
                leaves_to_explore.remove(best_split.leaf)

                # Add the new leaves to explore if they respect structural constraints
                if parent_depth + 1 < max_depth:
                    if len(left_indices) >= self.min_samples_split:
                        leaves_to_explore.append(best_split.leaf)
                    if len(right_indices) >= self.min_samples_split:
                        leaves_to_explore.append(n_leaves)

                # Now, increment the number of leaves
                n_leaves += 1
                # Increment the number of clusters if it did happen
                if best_split.left_target >= n_clusters and best_split.right_target >= n_clusters:
                    # Double star gain
                    n_clusters += 2
                elif best_split.left_target >= n_clusters or best_split.right_target >= n_clusters:
                    # Single star gain
                    n_clusters += 1

        self.labels_ = (Y @ Z).argmax(0)
        self.leaves_ = Z.argmax(0)

        return self

    def fit_predict(self, X, y=None):
        """Performs the KAURI algorithm by repeatedly choosing leaves, evaluating best gain and increasing the tree
        structure until structural limits or maximal gains are reached. Returns the assigned clusters to the data
        samples.

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
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.tree_.predict(X)

    def score(self, X, y=None):
        """
        Return the value of the GEMINI evaluated on the given test data. Note that this GEMINI is a special variation
        for the MMD-GEMINI with dirac distributions and hence may be different up to constants or factors of the actual
        GEMINI.

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
        y_pred = self.predict(X)
        kernel = self._compute_kernel(X, y)
        return gemini_objective(y_pred, kernel)


@constraint_params({
    "kauri_tree": [Kauri],
    "feature_names": ["array-like", None]
})
def print_kauri_tree(kauri_tree, feature_names=None):
    """
    Prints the binary tree structure of a trained KAURI tree.

    Parameters
    ----------
    kauri_tree: Kauri
        A Kauri instance that was trained

    feature_names: array of shape (n_features,) or None
        The name to use to describe the features. If set to None, a default print "X[:,i]" is proposed.
    """
    assert isinstance(kauri_tree, Kauri), f"The passed instance is not a KauriTree, got: {kauri_tree.__class__}"
    check_is_fitted(kauri_tree)
    if feature_names is not None:
        used_features = [x for x in kauri_tree.tree_.features if x is not None]
        assert len(feature_names) >= len(np.unique(used_features)), ("Fewer feature names than used "
                                                                     "features by the tree were provided")

    def print_node(node_id):
        current_depth = kauri_tree.tree_.depths[node_id]
        print("| " * current_depth, f"Node {node_id}", sep="")
        left_child = kauri_tree.tree_.children_left[node_id]
        right_child = kauri_tree.tree_.children_right[node_id]
        if left_child == -1:
            print("| " * current_depth, f"Cluster: {kauri_tree.tree_.target[node_id]}")
            return
        feature = kauri_tree.tree_.features[node_id]
        threshold = kauri_tree.tree_.thresholds[node_id]
        if feature_names is not None:
            feature_name = feature_names[feature]
        else:
            feature_name = f"X[:, {feature}]"
        print("| " * current_depth, "|=", f"{feature_name} <= {threshold}", sep="")
        print_node(left_child)
        print("| " * current_depth, "|=", f"{feature_name} > {threshold}", sep="")
        print_node(right_child)

    print_node(0)
