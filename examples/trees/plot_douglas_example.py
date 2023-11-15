"""
==================================================================
Building a differentiable unsupervised tree: DOUGLAS
==================================================================

This example shows how to use the Douglas tree for a dataset with few features.

The DOUGLAS model builds a differentiable tree by associating different constructed binnings of the data per feature
to clusters. The thresholds are learnt by GEMINI maximisation.
"""
from sklearn import datasets, metrics

from gemclus.gemini import MMDOvA
from gemclus.tree import Douglas

###########################################################################
# Load the dataset
# --------------------------------------------------------------

# %%data

iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

###########################################################################
# Create the douglas tree and fit it
# --------------------------------------------------------------

# %%douglas_learn

model = Douglas(n_clusters=3, gemini=MMDOvA(), max_iter=100, n_cuts=1)
y_pred_linear = model.fit_predict(X)

print("Score of model is: ", model.score(X))
print("ARI of model is: ", metrics.adjusted_rand_score(y, y_pred_linear))
