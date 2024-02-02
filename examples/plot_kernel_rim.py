"""
==============================================================
Comparative clustering of circles dataset with kernel change
==============================================================

We show here a simple dataset consisting in two centred circle that can be challenging for some clustering algorithms.
This dataset can be challenging for GEMINI as well, unless we change the kernel adequately.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import pairwise_kernels

from gemclus.linear import KernelRIM, RIM

#######################################
# Draw samples for the circle dataset
# -------------------------------------

# %%data

# We start by generating samples distributed on two circles
X, y = datasets.make_circles(n_samples=200, noise=0.05, factor=0.05, random_state=0)

# then normalise the data
#X = (X - np.mean(X, 0)) / np.std(X, ddof=0)

# # Have a look at it
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.axis("off")
# plt.ylim((-3, 3))
# plt.ylim((-3, 3))
# plt.show()

#############################
# Training clustering models
# ---------------------------

# %%clustering
model_kernel = KernelRIM(n_clusters=2, base_kernel="poly", max_iter=100, reg=0.05, verbose=True)
y_pred = model_kernel.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()