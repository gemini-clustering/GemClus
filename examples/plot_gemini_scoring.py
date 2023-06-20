"""
==============================
Scoring any model with GEMINI
==============================

We show in this example how we can score the prediction of another model using GEMINI. We do not seek to perform
clustering but rather to evaluate.
"""
import numpy as np
from sklearn import datasets, preprocessing, linear_model, naive_bayes
from gemclus import gemini

###########################################################################
# Load a simple real dataset
# --------------------------------------------------------------

# %%data

X, y = datasets.load_breast_cancer(return_X_y=True)
# Preprocess this dataset
X = preprocessing.RobustScaler().fit_transform(X)

###########################################################################
# Train two supervised models
# --------------------------------------------------------------
# We will train two different models on the breast cancer dataset

# %%trainingmodels

# The first model is a simple logistic regression with l2 penalty
clf1 = linear_model.LogisticRegression(random_state=0).fit(X, y)
p_y_given_x_1 = clf1.predict_proba(X)

# The second model is naive Bayes using Gaussian hypotheses on the data
clf2 = naive_bayes.GaussianNB().fit(X, y)
p_y_given_x_2 = clf2.predict_proba(X)

##########################################################################
# Scoring with GEMINI
# -----------------
# We can now score the clustering performances of both model with GEMINI.

# %%scoring

# Let's start with the WassersteinGEMINI (one-vs-all) and the Euclidean distance
wasserstein_scoring = gemini.WassersteinOvA(metric="euclidean")

# We need to precompute the affinity matching this Wasserstein (will be the Euclidean metric here)
affinity = wasserstein_scoring.compute_affinity(X)
clf1_score = wasserstein_scoring.evaluate(p_y_given_x_1, affinity)
clf2_score = wasserstein_scoring.evaluate(p_y_given_x_2, affinity)

print("Wasserstein OvA (Euclidean):")
print(f"\t=>Linear regression: {clf1_score:.3f}")
print(f"\t=>Naive Bayes: {clf2_score:.3f}")

##########################################################################
# Supervised Scoring with GEMINI
# -----------------
# By replacing the Euclidean distance for a label-informed distance
# we can obtain a supervised metric.

# %%supervisedscoring
# We now specify that the metric is precomputed instead
wasserstein_scoring = gemini.WassersteinOvA(metric="precomputed")

# So, we precompute a distance where samples have distance 0 if they share the same label, 1 otherwise
y_one_hot = np.eye(2)[y]
precomputed_distance = 1 - np.matmul(y_one_hot, y_one_hot.T)
clf1_score = wasserstein_scoring.evaluate(p_y_given_x_1, precomputed_distance)
clf2_score = wasserstein_scoring.evaluate(p_y_given_x_2, precomputed_distance)

print("Wasserstein OvA (Supervised):")
print(f"\t=>Linear regression: {clf1_score:.3f}")
print(f"\t=>Naive Bayes: {clf2_score:.3f}")
