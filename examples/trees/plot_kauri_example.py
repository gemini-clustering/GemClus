"""
==================================================================
Building an unsupervised tree with kernel-kmeans objective: KAURI
==================================================================

We show here how to obtain two different decision trees for clustering using two different kernels to accompanny
the KAURI method.

The KAURI model builds decision trees using gain metrics derived from the squared MMD-GEMINI which are equivalent
to KMeans optimisation.
"""
from sklearn import datasets, metrics
from gemclus.tree import Kauri, print_kauri_tree

###########################################################################
# Load the dataset
# --------------------------------------------------------------

# %%data

iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

###########################################################################
# Create a first tree using a linear kernel
# --------------------------------------------------------------

# %%linear_kernel

# Notice that we limit the depth of the tree for simplicity
linear_model = Kauri(max_clusters=3, kernel="linear", max_depth=3)
y_pred_linear = linear_model.fit_predict(X)

print("Score of model is: ", linear_model.score(X))

###########################################################################
# Create a second tree using an additive chi2 kernel
# --------------------------------------------------------------

# %%additive_kernel

additive_chi2_model = Kauri(max_clusters=3, kernel="additive_chi2", max_depth=3)
y_pred_additive_chi2 = additive_chi2_model.fit_predict(X)
print("Score of model is: ", additive_chi2_model.score(X))

###########################################################################
# Evaluate the performances of the model
# --------------------------------------------------------------

# %%performances

print("ARI of linear kernel: ", metrics.adjusted_rand_score(y,y_pred_linear))
print("ARI of additive chi2 kernel: ", metrics.adjusted_rand_score(y,y_pred_additive_chi2))

###########################################################################
# Visualise the tree structure
# --------------------------------------------------------------

# %%visualise_structure

print("Structure of the additive chi2 model")
print_kauri_tree(additive_chi2_model, iris["feature_names"])