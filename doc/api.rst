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
    :toctree: generated/gemini/
    :template: class.rst

    gemini.MMDGEMINI
    gemini.WassersteinGEMINI
    gemini.MI
    gemini.KLGEMINI
    gemini.TVGEMINI
    gemini.HellingerGEMINI

Clustering models
==================

Dense models
-------------

These models are based on standard distributions like the logistic regression or the one-hidden-layer neural network for
clustering.

.. autosummary::
   :toctree: generated/models/
   :template: class.rst

    linear.LinearModel
    linear.LinearMMD
    linear.LinearWasserstein
    linear.RIM
    mlp.MLPModel
    mlp.MLPMMD
    mlp.MLPWasserstein

Nonparametric models
--------------------

These models have parameters that are assigned to the data samples according to their indices. Consequently, the
parameters do not have any dependence on the location of the samples. Overall, these models can be used to model
any decision boundary and do not have hyper parameters. However, the underlying distribution cannot be used on
unseen samples for prediction.

.. autosummary::
   :toctree: generated/models/
   :template: class.rst

    nonparametric.CategoricalModel
    nonparametric.CategoricalMMD
    nonparametric.CategoricalWasserstein

Sparse models
--------------

These models can be trained to progressively remove features in the conditional cluster distribution. They are useful
for selecting a subset of features which may enhance interpretability of clustering.

.. autosummary::
   :toctree: generated/models/
   :template: class.rst

    sparse.SparseLinearModel
    sparse.SparseLinearMI
    sparse.SparseLinearMMD
    sparse.SparseMLPModel
    sparse.SparseMLPMMD

Tree models
------------

We propose clustering methods based on tree architectures. Thus rules are simultaneously constructed as the clustering
is learnt.

.. autosummary::
    :toctree: generated/models/
    :template: class.rst

    tree.Kauri
    tree.Douglas

The following functions are intended to help understanding the structure of the above models by printing their
inner rules.

.. autosummary::
    :toctree: generated/models/
    :template: function.rst
    
    tree.print_kauri_tree
    
    
    
Constraints
===========

This method aims at decorating the GEMINI models to give further guidance on the desired clustering.

.. autosummary::
    :toctree: generated/constraints/
    :template: function.rst
    
    add_mlcl_constraint

Dataset generation
===================

This package contains simple functions for generating synthetic datasets.

.. autosummary::
    :toctree: generated/data/
    :template: function.rst

    data.draw_gmm
    data.multivariate_student_t
    data.gstm
    data.celeux_one
    data.celeux_two
