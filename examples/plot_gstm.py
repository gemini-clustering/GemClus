"""
================================================================================================
Example of decision boundary map for a mixture of Gaussian and low-degree Student distributions
================================================================================================

This example is a retake from the experiment in the original GEMINI paper where we want to find the true clusters
in a mixture of Gaussian that incorporates a low-degree-of-freedom student t-distribution. Consequently, this
distribution generates sample that may seem like outliers if we are to expect only Gaussian distribution.

Unlike the paper, this example here is done with the `gemclus.linear.LinearWasserstein` instead of an MLP.
"""

from matplotlib import pyplot as plt
from gemclus.linear import LinearWasserstein
import numpy as np

##########################################################################
# Generate the data
# -----------------
#

#%%data
# Taking 200 samples, 1 degree of freedom and not-so-far apart means
N=200
alpha = 3
df = 1

# Generate first the multivariate Gaussian distribution
np.random.seed(0)

py = np.ones(4) / 4
means = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * alpha

y = np.random.multinomial(1, py, size=N).argmax(1)
proportions = np.bincount(y)
y.sort()

covariance = np.eye(2)

X = []
for k in range(3):
    X += [np.random.multivariate_normal(means[k], covariance, size=proportions[k])]

# Sample from the student t distribution
nx = np.random.multivariate_normal(np.zeros(2), covariance, size=proportions[-1])
u = np.random.chisquare(df, proportions[-1]).reshape((-1, 1))
x = np.sqrt(df / u) * nx + np.expand_dims(means[-1], axis=0)
X += [x]

X = np.concatenate(X, axis=0)

##########################################################################
# Train the model for clustering
# -------------------------------
#

#%%training

clf = LinearWasserstein(n_clusters=4, random_state=0)
y_pred = clf.fit_predict(X)


##########################################################################
# Final Clustering
# -----------------

#%%clustering

# Now, generate as well grid inputs to help drawing the decision boundary
x_vals = np.linspace(-10,10,num=50)
y_vals = np.linspace(-10,10,num=50)
xx,yy=np.meshgrid(x_vals,y_vals)
grid_inputs = np.c_[xx.ravel(),yy.ravel()]
zz = clf.predict(grid_inputs).reshape((50,50))

# Plot decision boundary with predictions on top
plt.contourf(xx,yy,zz,alpha=0.5,cmap="Blues")
plt.scatter(X[:,0],X[:,1],c=y_pred,cmap="Reds_r")

plt.xlim(-10,10)
plt.ylim(-10,10)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()
