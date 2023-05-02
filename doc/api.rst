####################
GEMCLUS API
####################

The GEMINI-clustering package currently contains simple MLP and logistic regression for all-feature clustering as well as
sparsity-constrained variants of these models.

.. currentmodule:: gemclus

Clustering models
==================

Dense models
-------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

    linear.LinearMMD
    linear.LinearWasserstein
    linear.RIM
    mlp.MLPMMD
    mlp.MLPWasserstein

Sparse models
--------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

    sparse.SparseLinearMMD
    sparse.SparseMLPMMD

Dataset generation
===================

.. autosummary::
    :toctree: generated/
    :template: function.rst

    data.draw_gmm
    data.multivariate_student_t
    data.gstm
    data.celeux_one
    data.celeux_two