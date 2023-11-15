.. title:: User guide : contents

.. _user_guide:

#####################################
User Guide
#####################################

Content of the package
=======================

Which GEMINIs are implemented
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, the `gemclus` package implements both the MMD and Wasserstein distances between the cluster distributions as
in the original paper. Both OvA and OvO implementations are present in all models. The OvO mode can be set in any
clustering model by adding :code:`ovo=True` in the constructor of a model.

We propose as well an extension of the base GEMINI algorithm with a sparsity-constrained model that adds a `Group-Lasso`
penalty to achieve joint feature selection and discriminative clustering. However, regarding this specific version we
chose so far to only implement it using the MMD-GEMINI because the Wasserstein distance seems to yield worst
performances so far.

The Wasserstein distance requires a distance function in the data space to compute. We directly propose all distances
available from :class:`sklearn.metrics.pairwise_distances`. In the same manner, we provide all kernels available
from :class:`sklearn.metrics.pairwise_kernels` for the MMD. By default, all GEMINIs use the Euclidean norm between
sample.

What discriminative distributions are available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We made the choice to force in this library the usage of 2-layers Multi-Layered-Perceptrons to keep the discriminative
distribution :math:`p_\theta(y|x)` flexible yet expressive enough in terms of decision boundaries. We provide as well
logistic regression models in case you want a simpler decision boundary.

For the sparse version of the algorithm, the architecture is inspired from `LassoNet <https://lassonet.ml/>`_ which
adds a linear skip connection between the inputs and clustering output. Note that there is as well a more simple
logistic regression model with group-lasso penalty to induce sparsity with :class:`gemclus.sparse.SparseLinearMMD`.

If you want to use another model, you can derive one of the :class:`gemclus._BaseMMD` or :class:`gemclus._BaseWasserstein`
classes and rewrite their hidden method :code:`_infer`, :code:`_get_weights` and :code:`_init_params`. An example
of extension is given `Here <auto_examples/plot_custom_model.html>`_

Basic examples
===============

We provide some basic examples in the `Example gallery <auto_examples/index.html>`_, including clustering of simple distribution
and how to perform feature selection using sparse models from :class:`gemclus.sparse`.