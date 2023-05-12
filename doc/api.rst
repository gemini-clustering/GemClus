####################
GEMCLUS API
####################

The GEMINI-clustering package currently contains simple MLP and logistic regression for all-feature clustering as well as
sparsity-constrained variants of these models.

.. currentmodule:: gemclus

Scoring with GEMINI
====================

The following classes implement the basic GEMINIs for scoring and evaluating any conditional distribution for
clustering.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    gemini.MMDOvA
    gemini.MMDOvO
    gemini.WassersteinOvA
    gemini.WassersteinOvO
    gemini.MI

Clustering models
==================

Dense models
-------------

These models are based on standard distributions like the logistic regression or the one-hidden-layer neural network for
clustering.

.. autosummary::
   :toctree: generated/
   :template: class.rst

    linear.LinearMMD
    linear.LinearWasserstein
    linear.RIM
    mlp.MLPMMD
    mlp.MLPWasserstein

Nonparametric models
--------------------

These models have parameters that are assigned to the data samples according to their indices. Consequently, the
parameters do not have any dependence on the location of the samples. Overall, these models can be used to model
any decision boundary and do not have hyper parameters. However, the underlying distribution cannot be used on
unseen samples for prediction.

.. autosummary::
   :toctree: generated/
   :template: class.rst

    nonparametric.CategoricalMMD
    nonparametric.CategoricalWasserstein

Sparse models
--------------

These models can be trained to progressively remove features in the conditional cluster distribution. They are useful
for selecting a subset of features which may enhance interpretability of clustering.

.. autosummary::
   :toctree: generated/
   :template: class.rst

    sparse.SparseLinearMMD
    sparse.SparseMLPMMD

Dataset generation
===================

This package contains simple functions for generating synthetic datasets.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    data.draw_gmm
    data.multivariate_student_t
    data.gstm
    data.celeux_one
    data.celeux_two