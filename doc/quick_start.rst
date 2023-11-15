#####################################
Quick start on gemclus
#####################################

We provide here a short description of the GEMINI clustering package and hints about what it can do or not.

.. note::
    For more details on the definition of GEMINI and its performances, please refer to `the original
    publication <https://openreview.net/pdf?id=0Oy3PiA-aDp>`_ by Ohl et al [1].

What is GEMINI
================

GEMINI stands for `Generalised Mutual Information`, a novel type of information theory score that can be used as an
objective to maximise to perform clustering. GEMINI consists in measuring the expectation of a distance :math:`D`
between custer distributions. For a set of clustering distributions :math:`p_\theta(x|y)`, GEMINI has two definitions.
The first one is the one-vs-all (OvA) which compares the cluster distribution to the data distribution:

.. math::

    \mathbb{E}_{y \sim p_\theta(y)} \left[ D(p_\theta(x|y) \| p(x))\right],

and the one-vs-one (OvO) version which compares two independently drawn cluster distributions:

.. math::

    \mathbb{E}_{y_1, y_2 \sim p_\theta(y)} \left[ D(p_\theta(x|y_1) \| p_\theta(x | y_2))\right].

The specificity of GEMINI is that it involves distances in which the Bayes Theorem can easily be performed to get
a tractable objective that we cane valuate using only clustering probabilities. Hence, models trained with GEMINI
are discriminative models :math:`p_\theta(y|x)` without any parametric assumption.

Doing discriminative clustering
===============================

The package respects the `scikit-learn` conventions for models API. Thus, doing clustering with the GEMINI looks like::

    # Import the model and a simple datasets
    from gemclus.mlp import MLPMMD
    from sklearn.datasets import load_iris
    X,y = load_iris(return_X_y=True)

    # Fit GEMINI clustering
    model = MLPMMD(n_clusters=3).fit(X)
    model.predict(X)

.. note::
    At the moment, and as reported in [1], GEMINI models may converge to using fewer clusters than asked in the models.
    It is thus a good practice to get models to run several times and get a good merge of the results.

For the details of the available models and GEMINI losses, you may check the `API reference <api.html>`_. Moreover, we
give additional hints on how to derive your own model from the base classes in the `User Guide <user_guide.html>`_.

Selecting features in clustering
==================================

We further propose an improvement of the GEMINI clustering to bring feature selection. This is mainly inspired from
[2] and was proposed in [3]. If you feel interested in feature selection, take a look at our
`sparse models <api.html#sparse-clustering-models>`_.

References
===========
.. [1] Ohl, L., Mattei, P.-A., Bouveyron, C., Harchaoui, W., Leclercq, M., Droit, A., & Precioso, F. (2022).
    `Generalised Mutual Information for Discriminative Clustering <https://openreview.net/pdf?id=0Oy3PiA-aDp>`_.
    In A. H. Oh, A. Agarwal, D. Belgrave, & K. Cho (Eds.), Advances in Neural Information Processing Systems.

.. [2] Lemhadri, I., Ruan, F., Abraham, L., & Tibshirani, R. (2021). `LassoNet: A Neural Network with Feature Sparsity
    <https://lassonet.ml/>`_. Journal of Machine Learning Research, 22(127), 1–29.

.. [3] Ohl, L., Mattei, P.-A., Bouveyron, C., Leclercq, M., Droit, A., & Precioso, F. (2023).
    `Sparse GEMINI for Joint Discriminative Clustering and Feature Selection <https://arxiv.org/abs/2302.03391>`_.
    doi:10.48550/ARXIV.2302.03391