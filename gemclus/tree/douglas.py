from functools import reduce
from numbers import Integral, Real

import numpy as np

from .._base_gemini import DiscriminativeModel
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted


class Douglas(DiscriminativeModel):
    """
    Implementation of the `DNDTs optimised using GEMINI leveraging apprised splits` tree algorithm. This model learns
    clusters by optimising learnable parameters to perform feature-wise soft-binnings and recombine those bins
    into a single cluster predictions. The parameters are optimised to maximise a generalised mutual information score.

    Parameters
    ----------

    n_clusters : int, default=3
        The number of clusters to form as well as the number of output neurons in the neural network.

    gemini: str, GEMINI instance or None, default="wasserstein_ova"
        GEMINI objective used to train this discriminative model. Can be "mmd_ova", "mmd_ovo", "wasserstein_ova",
        "wasserstein_ovo", "mi" or other GEMINI available in `gemclus.gemini.AVAILABLE_GEMINI`. Default GEMINIs
        involve the Euclidean metric or linear kernel. To incorporate custom metrics, a GEMINI can also
        be passed as an instance. If None, the GEMINI will be the MMD OvA.

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

    solver: {'sgd','adam'}, default='adam'
        The solver for weight optimisation.

        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimiser proposed by Kingma, Diederik and Jimmy Ba.

    learning_rate: float, default=1e-2
        The learning rate hyperparameter for the optimiser's update rule.

    verbose: bool, default=False
        Whether to print progress messages to stdout

    random_state: int, RandomState instance, default=None
        Determines random number generation for feature exploration.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    optimiser_: `AdamOptimizer` or `SGDOptimizer`
        The optimisation algorithm used for training depending on the chosen solver parameter.
    labels_: ndarray of shape (n_samples)
        The labels that were assigned to the samples passed to the :meth:`fit` method.
    n_iter_: int
        The number of iterations that the model took for converging.
    """

    _parameter_constraints: dict = {
        **DiscriminativeModel._parameter_constraints,
        "n_cuts": [Interval(Integral, 1, None, closed="left"), None],
        "feature_mask": [np.ndarray, None],
        "temperature": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, n_clusters=3, gemini="wasserstein_ova", n_cuts=1, feature_mask=None, temperature=0.1,
                 max_iter=100, batch_size=None, solver="adam", learning_rate=1e-2, verbose=False, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            gemini=gemini,
            max_iter=max_iter,
            learning_rate=learning_rate,
            solver=solver,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state
        )
        self.n_cuts = n_cuts
        self.feature_mask = feature_mask
        self.temperature = temperature

    def _leaf_binning(self, X, cut_points):
        n = len(cut_points)
        W = np.expand_dims(np.linspace(1, n + 1, n + 1, dtype=np.float64), axis=0)
        order = np.argsort(cut_points)
        sorted_cut_points = cut_points[order]
        b = np.cumsum(np.concatenate([np.zeros(1), -sorted_cut_points])).reshape((1, -1))

        logits = X @ W + b

        return softmax(logits / self.temperature), order

    def _merge_leaf(self, leaf_res1, leaf_res2):
        # Compute feature-wise kronecker product
        product = np.einsum("ij,ik->ijk", leaf_res1, leaf_res2)

        # reshape to 2d
        return product.reshape((-1, np.prod(product.shape[1:])))

    def _infer(self, X, retain=True):
        leaf_binning = lambda z: self._leaf_binning(X[:, z[0]:z[0] + 1], z[1])
        cut_iterator = map(leaf_binning, self.cut_points_list_)

        all_binnings_results = list(cut_iterator)
        all_binnings = [x[0] for x in all_binnings_results]
        all_orders = [x[1] for x in all_binnings_results]

        leaf = reduce(self._merge_leaf, all_binnings)

        y_pred = leaf @ self.leaf_scores_

        if retain:
            self._leaf = leaf
            self._all_orders = all_orders
            self._all_binnings = all_binnings

        return softmax(y_pred)

    def _init_params(self, random_state, X=None):
        # Create the parameters
        if self.feature_mask is None:
            self.cut_points_list_ = [(i, random_state.normal(size=(self.n_cuts,))) for i in range(X.shape[1])]
            num_leaf = int((self.n_cuts + 1) ** X.shape[1])
        else:
            if len(self.feature_mask) != X.shape[1]:
                raise ValueError("The boolean feature mask must have as much entries as the number of features")
            self.cut_points_list_ = [(i, random_state.normal(size=self.n_cuts, )) for i in range(X.shape[1])
                                     if self.feature_mask[i]]
            num_leaf = int((self.n_cuts + 1) ** len(self.cut_points_list_))

        if self.verbose:
            print(f"Total will be {num_leaf} values per sample")
        self.leaf_scores_ = random_state.normal(size=(num_leaf, self.n_clusters))

    def _compute_grads(self, X, y_pred, gradient):
        # Start by the backprop through the softmax
        y_pred_grad = y_pred * (gradient - (y_pred * gradient).sum(1, keepdims=True))

        # Then backprop through the final matrix multiplication
        binning_backprop = y_pred_grad @ self.leaf_scores_.T
        leaf_score_backprop = self._leaf.T @ y_pred_grad

        # Store the update corresponding to the leaf score, negate for maximisation instead of minimisation
        updates = [-leaf_score_backprop]

        # Then, compute all feature kronecker derivatives
        axes_for_reshape = tuple([-1] + [len(x[1]) + 1 for x in self.cut_points_list_])
        binning_backprop = binning_backprop.reshape(axes_for_reshape)
        # We must multiply the binning gradient by all binnings
        binning_backprop *= self._leaf.reshape(axes_for_reshape)

        # Compute individual cut points backprop
        for i, (_, cut_points) in enumerate(self.cut_points_list_):
            axes_for_sum = tuple([1 + j for j in range(len(self.cut_points_list_)) if i != j])
            softmax_grad = binning_backprop.sum(axes_for_sum) / self._all_binnings[i]

            bin_grad = self._all_binnings[i] * (
                    softmax_grad - (self._all_binnings[i] * softmax_grad).sum(1, keepdims=True))  # Shape Nx(d+1)
            bin_grad /= self.temperature

            # Gradient is directly on the bias, so we only need to do the cumsum backprop after summing on
            # all samples

            # We remove the gradient on the constant [0]
            bias_grad = bin_grad.sum(0)[1:]
            cumsum_grad = -np.cumsum(bias_grad[::-1])[::-1]

            # Take the order back
            cut_grad = cumsum_grad[np.argsort(self._all_orders[i])]

            # Apply update rule and negate for maximisation instead of minimisation
            updates += [-cut_grad]

        return updates

    def _get_weights(self):
        return [self.leaf_scores_] + list(map(lambda x: x[1], self.cut_points_list_))

    def find_active_points(self, X):
        """
        Calculates the list of cut points that are considered as active for a Douglas tree and some data X. A cut point
        is active if its value falls within the bounds of its matching feature.

        Active points can be used for finding features that actively contributed to the clustering.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        active_points: List
            A list containing the integer indices of features for which the Douglas model has active cut points
        """

        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] < len(self.cut_points_list_):
            raise ValueError("The passed data has fewer features than the number of cut points expected for the "
                             "Douglas model")
        active_points = []

        for (feature_index, cut_points) in self.cut_points_list_:
            feature = X[:, feature_index]
            min_threshold = cut_points.min()
            max_threshold = cut_points.max()

            # Check of the cut point lists having at least one threshold falling within bounds of the feature
            if not (np.all(feature <= min_threshold) or np.all(feature >= max_threshold)):
                active_points += [feature_index]

        return active_points
