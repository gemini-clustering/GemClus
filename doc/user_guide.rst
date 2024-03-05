.. title:: User guide : contents

.. _user_guide:

#####################################
User Guide
#####################################

Content of the package
=======================

Which GEMINIs are implemented
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All GEMINIs from the initial work are available: MMD and Wasserstein distances are present for geometrical
considerations, as well as Kullback-Leibler divergence, Total Variation distance and squared Hellinger distance.
Both OvA and OvO implementations are present in all models. The OvO mode can be set in most
clustering model by adding :code:`ovo=True` in the constructor of a model.

Some models propose readily integrated GEMINIs, but it is also possible to set a custom GEMINI for some models.

The Wasserstein distance requires a distance function in the data space to compute. We directly propose all distances
available from :class:`sklearn.metrics.pairwise_distances`, with the Euclidean distance by default.
In the same manner, we provide all kernels available from :class:`sklearn.metrics.pairwise_kernels` for the MMD, with
the linear kernel by default. For both GEMINIs, it is possible as well to involve a precomputed distance or kernel of
your own that must be then passed to the GEMINI.

What discriminative distributions are available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We propose several clustering distributions depending on the purpose:

+ Logistic regressions
+ 2-layer Multi-Layered-Perceptrons with ReLU activations
+ Decision trees (only compatible with the MMD GEMINI)
+ Differentiable trees

For the logistic regression and MLP, we also propose sparse versions to achieve feature selection along clustering.
The sparse architecture of the MLP is inspired from `LassoNet <https://lassonet.ml/>`_ [1]_ which
adds a linear skip connection between the inputs and clustering output.

We also include other models taken from the litterature that fits the scope of discriminative clustering with mutual
information, e.g. the regularized mutual information (RIM): :class:`gemclus.linear.RIM` [2]_.

If you want to use another model, you can derive the :class:`gemclus.DiscriminativeModel`
class and rewrite its hidden methods :code:`_infer`, :code:`_get_weights`, :code:`_init_params` and
:code:`_compute_grads`. An example
of extension is given `Here <auto_examples/_general/plot_custom_model.html>`_

Basic examples
===============

We provide some basic examples in the `Example gallery <auto_examples/index.html>`_, including clustering of simple distribution
and how to perform feature selection using sparse models from :class:`gemclus.sparse`.

.. [1] Lemhadri, I., Ruan, F., Abraham, L., & Tibshirani, R. (2021). `LassoNet: A Neural Network with Feature Sparsity
    <https://lassonet.ml/>`_. Journal of Machine Learning Research, 22(127), 1–29.

.. [2] Krause, A., Perona, P., & Gomes, R. (2010).
    `Discriminative Clustering by Regularized Information Maximization <https://proceedings.neurips.cc/paper_files/paper/2010/file/42998cf32d552343bc8e460416382dca-Paper.pdf>`_.
    In J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, & A. Culotta (Eds.), Advances in Neural Information
    Processing Systems (Vol. 23).