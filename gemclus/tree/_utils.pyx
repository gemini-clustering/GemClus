import numpy as np
cimport numpy as np

cdef class Split:

    cdef readonly np.float64_t gain, threshold # FIXME: perhaps double
    cdef readonly bint is_categorical
    cdef readonly np.int64_t leaf, left_target, right_target, feature

    def __init__(self, np.float64_t gain, np.int64_t leaf, np.int64_t left_target, np.int64_t right_target,
                    np.int64_t feature, np.float64_t threshold, bint is_categorical):
        self.gain = gain
        self.left_target = left_target
        self.right_target = right_target
        self.is_categorical = is_categorical
        self.feature = feature
        self.threshold = threshold
        self.leaf = leaf

    cpdef set_decision(self, np.int64_t feature, np.float64_t threshold):
        self.feature = feature
        self.threshold = threshold

    cpdef set_targets(self, np.int64_t left_target, int right_target):
        self.left_target = left_target
        self.right_target = right_target

    cpdef set_gain(self, np.float64_t gain):
        self.gain = gain

    cpdef set_leaf(self, np.int64_t leaf):
        self.leaf = leaf

    def __gt__(self, Split other_split):
        return self.gain >= other_split.gain


# Objective definitions
cdef np.float64_t kernel_stock(np.float64_t[:,:] kernel,
                                np.int64_t[:] indices_a,
                                np.int64_t[:] indices_b=None):
    """
    Computes the stock (i.e. grand-sum) of a kernel for given set of indices

    :param kernel: [array of shape (nxn)] a psd kernel
    :param indices_a: [array of shape m<n] indices on which to sum the kernel.
    :param indices_b: [array of shape k<n] second set of indices on which to sum the kernel. If not provided, then
        the sum is done over indices_a x indices_a.

    :return: [float] the sum of the kernel elements
    """
    cdef Py_ssize_t n = kernel.shape[0]
    cdef Py_ssize_t len_a = indices_a.shape[0]
    cdef Py_ssize_t len_b
    if indices_b is not None:
        len_b = indices_b.shape[0]
    else:
        len_b = 0

    cdef np.float64_t tmp = 0
    cdef Py_ssize_t i,j

    if indices_b is None:
        for i in range(len_a):
            for j in range(len_a):
                tmp += kernel[indices_a[i], indices_a[j]]
    else:
        for i in range(len_a):
            for j in range(len_b):
                tmp += kernel[indices_a[i],indices_b[j]]

    return tmp


# Define the objective
def gemini_objective(np.int64_t[:] y_pred, np.float64_t[:,:] kernel):
    """
    Computes an objective derived from the MMD-GEMINIs

    :param y_pred: [array of shape n] cluster assignments of each sample (1<=k<=K)
    :param kernel: [array of shape nxn] a psd kernel

    :return: \sum_{k=1}^K 1/|C_k| * kernel_stock(kernel, C_k)
    """
    cdef np.float64_t score = 0
    cdef np.ndarray[np.int64_t, ndim=1] indices

    for value in np.unique(y_pred):
        indices, = np.where(y_pred == value)
        score += kernel_stock(kernel, indices) / len(indices)

    return score


cdef Split compute_all_splits(Split best_split,
                               np.float64_t sl_square,
                               np.float64_t sr_square,
                               np.float64_t leaf_square,
                               np.float64_t[:] sl_clusters,
                               np.float64_t[:]sr_clusters,
                               np.int64_t[:] cluster_sizes,
                               np.float64_t[:,:] gamma,
                               np.float64_t[:,:] omega,
                               np.int64_t n_leaf,
                               Py_ssize_t n_clusters,
                               np.int64_t K_max,
                               np.int64_t k,
                               np.int64_t leaf_id,
                               np.int64_t split_size,
                               np.int64_t feature_id,
                               np.float64_t threshold):
    """

    :param best_split: [Split instance] the current best split found
    :param sl_square: [float] the kernel stock of the left split
    :param sr_square: [float] the kernel stock of the right split
    :param leaf_square: [float] the kernel stock of the leaf that we are splitting
    :param sl_clusters: [array of shape K] the kernel stock between the left split and all clusters
    :param sr_clusters: [array of shape K] the kernel stock between the right split and all clusters
    :param cluster_sizes: [array of shape K] the number of samples per cluster
    :param gamma: [array of shape KxK] the kernel stock between all pairs of clusters
    :param omega: [array of shape Kxn] the kernel stock per cluster per sample
    :param n_leaf: [int] the number of samples in the current leaf
    :param n_clusters: [int] the number of samples in the cluster to which the current leaf belongs
    :param K_max: [int] the maximal number of clusters
    :param k: [int] the cluster id of the current leaf
    :param leaf_id: [int] the number of the current leaf
    :param split_size: [int] the number of samples going to the left split
    :param feature_id: [int] the id of the feature on which a threshold for splitting is considered
    :param threshold: [float] the value of the threshold
    :return:
    """
    cdef np.int64_t delta_size, left_size, right_size
    cdef np.float64_t leaf_star, split_star, sl_sr, double_star_gain
    cdef Py_ssize_t k_prime = 0
    cdef Py_ssize_t top_k_left, second_k_left, top_k_right, second_k_right
    cdef np.float64_t top_gain_left, second_gain_left, top_gain_right, second_gain_right
    cdef np.int64_t delta_size_k, delta_size_k_prime
    cdef np.float64_t left_switch, right_switch
    cdef np.float64_t leaf_cluster, refurbish, corrective_term
    cdef Py_ssize_t k_left, k_right

    # Star gains
    if n_clusters < K_max - 1 and n_leaf != cluster_sizes[k]:
        # Double star gain

        # Compute leaf self kernel
        delta_size = cluster_sizes[k] - n_leaf
        # First the leaf leaves the cluster
        leaf_star = leaf_square * (1 / n_leaf + 1 / delta_size) + gamma[k, k] * (
                1 / delta_size - 1 / cluster_sizes[k])
        leaf_star -= 2 * omega[k, feature_id] / delta_size

        delta_size = n_leaf - split_size
        split_star = sl_square * (1 / split_size + +1 / delta_size) + leaf_square * (
                1 / delta_size - 1 / n_leaf)
        # sigma(Sl\times N) + sigma(Sr\times N) = sigma(N^2)
        # sigma(Sl, sl) + sigma(sr,sr) - 2sigma(N^2) = 2* leftover
        sl_sr = (leaf_square - sl_square - sr_square) / 2
        split_star -= (sl_square + sl_sr) / delta_size

        double_star_gain = split_star + leaf_star
        if double_star_gain > best_split.gain:
            best_split.set_gain(double_star_gain)
            best_split.set_targets(n_clusters, n_clusters + 1)
            best_split.set_decision(feature_id, threshold)
            best_split.set_leaf(leaf_id)

    if n_clusters < K_max:
        # Single star gain

        # We directly apply the formula
        left_size = split_size
        delta_size = cluster_sizes[k] - left_size
        left_star = sl_square * (1 / left_size + 1 / delta_size) + gamma[k, k] * (
                1 / delta_size - 1 / cluster_sizes[k])
        left_star -= 2 * sl_clusters[k] / delta_size

        right_size = n_leaf - left_size
        delta_size = cluster_sizes[k] - right_size
        right_star = sr_square * (1 / right_size + 1 / delta_size) + gamma[k, k] * (
                1 / delta_size - 1 / cluster_sizes[k])
        right_star -= 2 * sr_clusters[k] / delta_size

        # Retain the best
        if left_star > best_split.gain or right_star > best_split.gain:
            if left_star > right_star:
                best_split.set_gain(left_star)
                best_split.set_targets(n_clusters, k)
            else:
                best_split.set_gain(right_star)
                best_split.set_targets(k, n_clusters)
            best_split.set_decision(feature_id, threshold)
            best_split.set_leaf(leaf_id)

    if n_clusters >= 2:
        # Switch allowed

        # Initialise variables for refurbishing
        top_gain_left, second_gain_left, top_k_left, second_k_left = -np.inf, -np.inf, -1, -1
        top_gain_right, second_gain_right, top_k_right, second_k_right = -np.inf, -np.inf, -1, -1

        for k_prime in range(n_clusters):
            if k == k_prime:
                continue

            # Compute switch equation
            left_size = split_size
            delta_size_k = cluster_sizes[k] - left_size
            delta_size_k_prime = cluster_sizes[k_prime] + left_size
            left_switch = sl_square * (1 / delta_size_k_prime + 1 / delta_size_k)
            left_switch += gamma[k, k] * (1 / delta_size_k - 1 / cluster_sizes[k])
            left_switch -= 2 * sl_clusters[k] / delta_size_k
            left_switch += gamma[k_prime, k_prime] * (1 / delta_size_k_prime - 1 / cluster_sizes[k_prime])
            left_switch += 2 * sl_clusters[k_prime] / delta_size_k_prime

            right_size = n_leaf - left_size
            delta_size_k = cluster_sizes[k] - right_size
            delta_size_k_prime = cluster_sizes[k_prime] + right_size
            right_switch = sr_square * (1 / delta_size_k_prime + 1 / delta_size_k)
            right_switch += gamma[k, k] * (1 / delta_size_k - 1 / cluster_sizes[k])
            right_switch -= 2 * sr_clusters[k] / delta_size_k
            right_switch += gamma[k_prime, k_prime] * (1 / delta_size_k_prime - 1 / cluster_sizes[k_prime])
            right_switch += 2 * sr_clusters[k_prime] / delta_size_k_prime

            # Update top switch gains per left/right split
            if left_switch >= top_gain_left:
                top_gain_left, second_gain_left = left_switch, top_gain_left
                top_k_left, second_k_left = k_prime, top_k_left
            elif left_switch >= second_gain_left:
                second_gain_left = left_switch
                second_k_left = k_prime
            if right_switch >= top_gain_right:
                top_gain_right, second_gain_right = right_switch, top_gain_right
                top_k_right, second_k_right = k_prime, top_k_right
            elif left_switch >= second_gain_right:
                second_gain_right = right_switch
                second_k_right = k_prime

            # Update best gain
            if left_switch >= best_split.gain or right_switch >= best_split.gain:
                if left_switch > right_switch:
                    best_split.set_gain(left_switch)
                    best_split.set_targets(k_prime, k)
                else:
                    best_split.set_gain(right_switch)
                    best_split.set_targets(k, k_prime)
                best_split.set_decision(feature_id, threshold)
                best_split.set_leaf(leaf_id)

        if n_clusters >= 3 and n_leaf != cluster_sizes[k]:
            # Refurbish allowed

            # Compute the corrective term
            leaf_cluster = sl_clusters[k] + sr_clusters[k]  # sigma(NxCk) = sigma(SlxCk)+sigma(SrxCk)
            corrective_term = (gamma[k, k] + leaf_square - 2 * leaf_cluster) / (cluster_sizes[k] - n_leaf)
            corrective_term += gamma[k, k] / cluster_sizes[k]
            corrective_term -= (gamma[k, k] + sl_square - 2 * sl_clusters[k]) / (cluster_sizes[k] - split_size)
            corrective_term -= (gamma[k, k] + sr_square - 2 * sr_clusters[k]) / (
                    cluster_sizes[k] - n_leaf + split_size)

            # Choose the best pair of top switches

            if top_k_left != top_k_right:
                refurbish = top_gain_left + top_gain_right
                k_left, k_right = top_k_left, top_k_right
            else:
                if top_gain_left + second_gain_right > top_gain_right + second_gain_left:
                    refurbish = top_gain_left + second_gain_right
                    k_left, k_right = top_k_left, second_k_right
                else:
                    refurbish = top_gain_right + second_gain_left
                    k_left, k_right = second_k_left, top_k_right

            if refurbish + corrective_term > best_split.gain:
                best_split.set_gain(refurbish + corrective_term)
                best_split.set_targets(k_left, k_right)
                best_split.set_decision(feature_id, threshold)
                best_split.set_leaf(leaf_id)

def find_best_split(np.ndarray[np.float64_t, ndim=2] kernel, np.ndarray[np.float64_t, ndim=2] X,
                    np.ndarray[np.int64_t, ndim=1] leaves_to_explore, np.ndarray[np.int64_t, ndim=2] Y,
                    np.ndarray[np.int64_t, ndim=2] Z, Py_ssize_t n_clusters, np.int64_t K_max, np.int64_t n_leaves,
                    np.int64_t min_leaf, np.ndarray[Py_ssize_t, ndim=1] feature_subset) -> Split:

    """
    Given a set of leaf assignment, this function looks for the optimal leaf for creating the perfect split.

    :param kernel: [array of shape nxn] the psd kernel
    :param X: [array of shape nxd] the data to cluster
    :param leaves_to_explore [array of shape M] the list of leaf indices in which node exploration and split is allowed
    :param Y [array of shape KxL] the binary matrix describing assignments of leaves to clusters
    :param Z [array of shape Lxn] the binary matrix describing assignments of samples to leaves
    :param n_clusters [int] the number of clusters that are currently present in the model
    :param K_max [int] the maximal number of clusters allowed
    :param n_leaves [int] the number of leaves that are currently present in the tree
    :param min_leaf: [int], the minimal number of samples that must remain in a leaf
    :param feature_subset, for controlling the randomness of the process

    :return:
        - best_split [Split instance] a Split object summarising the best chosen split with leaf location, threshold,
            feature and cluster targets of left and right children.
    """
    cdef np.float64_t[:,:] Lambda, omega, gamma
    cdef Split best_split= Split(0, -1, -1, -1, -1, 0, False)
    cdef Py_ssize_t[:] leaf_indices
    cdef Py_ssize_t k, n_leaf, j, feature, l_split, l_prime
    cdef np.float64_t[:] subset_X
    cdef Py_ssize_t[:]  ordering, nu
    cdef np.float64_t sl_square, sr_square, leaf_square, alpha, beta
    cdef np.float64_t[:] sl_clusters, sr_clusters

    cdef Py_ssize_t a,b


    Lambda = np.matmul(Z[:n_leaves], kernel)
    omega = np.matmul(Y[:n_clusters, :n_leaves], Lambda)
    gamma = np.matmul(np.matmul(omega, np.transpose(Z[:n_leaves])), np.transpose(Y[:n_clusters, :n_leaves]))

    cluster_sizes = np.dot(Y, np.sum(Z, axis=1))

    for j in leaves_to_explore:
        leaf_indices,  = np.nonzero(Z[j])
        k = np.argmax(Y[:, j])
        n_leaf = len(leaf_indices)

        for feature in feature_subset:
            subset_X = np.zeros(n_leaf)
            for a in range(n_leaf):
                subset_X[a] = X[leaf_indices[a], feature] # Explicit loop for array indexing in cython

            ordering = np.argsort(subset_X)
            nu = np.zeros(n_leaf, dtype=np.intp)
            for a in range(n_leaf):
                nu[a] = leaf_indices[ordering[a]] # Explicit loop

            # Initialise variables for iterative search

            sl_square = 0
            sr_square = 0
            for a in range(n_leaf):
                sr_square += Lambda[j, leaf_indices[a]]

            leaf_square = sr_square
            sl_clusters = np.zeros(n_clusters) # originally omega.shape[0]
            sr_clusters = np.zeros(n_clusters)
            for a in range(n_leaf):
                for b in range(n_clusters):
                    sr_clusters[b] += omega[b, leaf_indices[a]]
            # sr_clusters = omega[:, leaf_indices].sum(1)

            for l_split in range(n_leaf -1):
                alpha, beta = 0, 0
                for l_prime in range(n_leaf):
                    if l_prime < l_split:
                        alpha += kernel[nu[l_split], nu[l_prime]]
                    elif l_prime > l_split:
                        beta += kernel[nu[l_split], nu[l_prime]]

                sl_square += 2 * alpha + kernel[nu[l_split], nu[l_split]]
                sr_square -= 2 * beta + kernel[nu[l_split], nu[l_split]]

                for a in range(n_clusters):
                    sl_clusters[a] += omega[a, nu[l_split]] # [:, nu[l_split]]
                    sr_clusters[a] -= omega[a, nu[l_split]]

                if l_split < (min_leaf - 1) or l_split > n_leaf - min_leaf - 1:
                    # We must guarantee a certain number of samples remaining in the leaves
                    continue

                if X[nu[l_split], feature] == X[nu[l_split+1], feature]:
                    # If two consecutive samples have the same feature value
                    # Both must be taken into account for the split
                    # Since the rules of the tree will necessarily keep those sample together
                    continue

                # Now, explore all possible splits
                compute_all_splits(best_split, sl_square, sr_square, leaf_square, sl_clusters, sr_clusters,
                                   cluster_sizes, gamma, omega, n_leaf, n_clusters, K_max, k, j, l_split + 1,
                                   feature, X[nu[l_split], feature])

    return best_split
