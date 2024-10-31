.. title:: User guide : contents

.. _user_guide:

#####################################
User Guide
#####################################


Discriminative clustering
=========================

Definition
----------

Clustering is the art of separating data samples :math:`x` into :math:`K` groups called *clusters*.

We take the word discriminative in the sense of Minka [2]_. In the context of clustering, this means that do not
set any hypotheses on the data distribution and only seek to cluster data by directly designing a clustering
distribution:

.. math::
    p_\theta(y \mid x),

where :math:`y` is the cluster to identify and :math:`\theta` are the parameters of the discriminative distribution.

Choosing a model
----------------

Any function that takes the data as input and returns a vector in a :math:`K-1`-simplex can be used as a discriminative
clustering distribution. For example, if we believe that the data is linearly separable, we can use a logistic
regression:

.. math::
    p_\theta (y = k \mid x) \propto \langle w_k, x \rangle + b_k.

Then, the parameters to learn are :math:`\theta = \{w_k, b_k\}_{k=1}^K`.

If we want more complex boundaries, any neural network that finishes with a softmax activation can be used. In other
words, the choice of the type of decision boundary should guide the choice of clustering distribution.

The GEMINI approach
-------------------

Learning parameters :math:`\theta` in discriminative clustering is challenging because the absence of hypothesis on the
data distribution prevents us from calculating any likelihood.

In 1991, Bridle, Heading and MacKay [1]_ proposed to optimise the parameters such that they maximise mutual information:

.. math::
    \mathcal{I} = \mathbb{E}_{y\sim p_\theta(y)} \left[D_\text{KL} (p_\theta(x\mid y) \| p_\text{data}(x))\right].

Mutual information has then stayed an essential component of discriminative clustering models. By comparison with
classification contexts, it is an objective function we can use independently of the form taken by the
model :math:`p_\theta(y\mid x)`.

The generalised mutual information (GEMINI) is an extension of mutual information that replaces the Kullback-Leibler
divergence :math:`D_\text{KL}` by any other statistical distance :math:`D`. It comes with two different versions.
The first approach, named *one-vs-all* (OvA) seeks to discriminate the individual cluster distributions from the data
distribution:

.. math::
    \mathbb{E}_{y \sim p_\theta(y)} \left[ D(p_\theta(x|y) \| p(x))\right].


The second approach is the *one-vs-one* (OvO) version that compares two independently drawn cluster distributions:

.. math::

    \mathbb{E}_{y_1, y_2 \sim p_\theta(y)} \left[ D(p_\theta(x|y_1) \| p_\theta(x | y_2))\right].

The specificity of GEMINI is that it involves distances in which the Bayes Theorem can easily be performed to get
a tractable objective that we cane valuate using only clustering probabilities. Hence, models trained with GEMINI
are discriminative models :math:`p_\theta(y|x)` without any parametric assumption.

Note that mutual information and the K-Means loss are special cases of GEMINI.


Extending / Regularising models
-------------------------------

Owing to the decoupling between the choice of the clustering model and the objective function to learn it, it is
possible to add regularisation that constraint the model :math:`p_\theta(y\mid x)`. For example, we propose to
add :math:`\ell_2` penalty on logistic regressions in :class:`gemclus.linear.RIM`, taken from Krause, Perona and
Gomes [3]_.

We also propose models that incorporate feature selection using group-lasso penalty, inspired from LassoNet.
Selecting feature can be interesting in the context of clustering for helping interpretation of clusters.

Content of the package
=======================

Which GEMINIs are implemented
------------------------------

All GEMINIs from our initial work are available [9]_: MMD and Wasserstein distances are present for geometrical
considerations, as well as Kullback-Leibler divergence, Total Variation distance and squared Hellinger distance.
Both OvA and OvO implementations are present in all models. The OvO mode can be set in most
clustering model by adding :code:`ovo=True` in the constructor of a model.

Some models propose readily integrated GEMINIs, but it is also possible to set a custom GEMINI for some models.

The Wasserstein distance requires a distance function in the data space to compute. We directly propose all distances
available from :class:`sklearn.metrics.pairwise_distances`, with the Euclidean distance by default.
In the same manner, we provide all kernels available from :class:`sklearn.metrics.pairwise_kernels` for the MMD, with
the linear kernel by default. For both GEMINIs, it is possible as well to involve a precomputed distance or kernel of
your own that must be then passed to the GEMINI.

All loss functions we propose are located in the module :class:`gemclus.gemini`.

What discriminative distributions are available
-----------------------------------------------

We propose several clustering distributions depending on the purpose:

+ Logistic regressions, kernel regressions in :class:`gemclus.linear`
+ 2-layer Multi-Layered-Perceptrons with ReLU activations in :class:`gemclus.mlp`
+ Decision trees (only compatible with the MMD GEMINI) in :class:`gemclus.tree`
+ Differentiable trees in :class:`gemclus.tree`
+ Sparse models in :class:`gemclus.sparse`
+ Nonparametric models in :class:`gemclus.nonparametric`

The sparse models are logistic regressions and MLP. Sparse means that we achieve feature selection along clustering.
The sparse architecture of the MLP is inspired from `LassoNet <https://lassonet.ml/>`_ [8]_ which
adds a linear skip connection between the inputs and clustering output. These sparse versions are located

If you want to use another model, you can derive the :class:`gemclus.DiscriminativeModel`
class and rewrite its hidden methods :code:`_infer`, :code:`_get_weights`, :code:`_init_params` and
:code:`_compute_grads`. An example
of extension is given `here <auto_examples/_general/plot_custom_model.html>`_


A summary of what is implemented
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We hope that GemClus will keep on growing. We seek to implement methods and datasets that can be relevant in
discriminative clustering.

.. list-table:: Models
    :widths: 60 40
    :header-rows: 1

    * - Class/Function
      - Source paper
    * - :class:`gemclus.linear.RIM`, :class:`gemclus.linear.KernelRIM`
      - Krause, Perona and Gomes [3]_
    * - :class:`gemclus.sparse.SparseLinearMI`
      - França, Rizzo and Vogelstein [5]_
    * - :class:`gemclus.tree.Kauri`
      - Ohl et al [11]_
    * - :class:`sparse.SparseMLPModel`
      - Ohl et al [10]_, Lemhadri et al [8]_
    * - :class:`sparse.SparseLinearModel`
      - Ohl et al [10]_

.. list-table:: Objective functions
    :widths: 60 40
    :header-rows: 1

    * - Class/Function
      - Source paper
    * - :class:`gemclus.gemini.WassersteinGEMINI`, :class:`gemclus.gemini.TVGEMINI`,
        :class:`gemclus.gemini.HellingerGEMINI`
      - Ohl et al [9]_
    * - :class:`gemclus.gemini.MMDGEMINI`
      - Ohl et al [9]_, [7]_
    * - :class:`gemclus.gemini.MI`
      - Bridle, Heading and MacKay [1]_
    * - :class:`gemclus.gemini.ChiSquareGEMINI`
      - Sugiyama et al [4]_


.. list-table:: Dataset
    :widths: 60 40
    :header-rows: 1

    * - Class/Function
      - Source paper
    * - :class:`gemclus.data.gstm`
      - Ohl et al [9]_
    * - :class:`gemclus.data.celeux_one`, :class:`gemclus.data.celeux_two`
      - Celeux et al [6]_



Basic examples
===============

We provide some basic examples in the `Example gallery <auto_examples/index.html>`_, including clustering of simple
distribution and how to perform feature selection using sparse models from :class:`gemclus.sparse`.

References
==========

.. [1] Bridle, J., Heading, A., & MacKay, D. (1991). `Unsupervised Classifiers, Mutual Information and 'Phantom
    Targets <https://proceedings.neurips.cc/paper/1991/hash/a8abb4bb284b5b27aa7cb790dc20f80b-Abstract.html>`_.
    Advances in Neural Information Processing Systems, 4.

.. [2] Minka, T. (2005). `Discriminative models, not discriminative training
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2005-144.pdf>`_.
    Technical Report MSR-TR-2005-144, Microsoft Research.

.. [3] Krause, A., Perona, P., & Gomes, R. (2010).
    `Discriminative Clustering by Regularized Information Maximization
    <https://proceedings.neurips.cc/paper_files/paper/2010/file/42998cf32d552343bc8e460416382dca-Paper.pdf>`_.
    In J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, & A. Culotta (Eds.), Advances in Neural Information
    Processing Systems (Vol. 23).

.. [4] Sugiyama, M., Yamada, M., Kimura, M., & Hachiya, H. (2011). `On Information-Maximization Clustering: Tuning
    Parameter Selection and Analytic Solution <http://www.icml-2011.org/papers/61_icmlpaper.pdf>`_. In Proceedings of
    the 28th International Conference on Machine Learning (ICML-11) (pp. 65-72).

.. [5] Kong, Y., Deng, Y., & Dai, Q. (2014). `Discriminative Clustering and Feature Selection for Brain MRI Segmentation
    <https://ieeexplore.ieee.org/abstract/document/6935074>`_. IEEE Signal Processing Letters, 22(5), 573-577.

.. [6] Celeux, G., Martin-Magniette, M. L., Maugis-Rabusseau, C., & Raftery, A. E. (2014). `Comparing Model Selection
    and Regularization Approaches to Variable Selection in Model-Based Clustering
    <http://www.numdam.org/item/JSFS_2014__155_2_57_0/>`_. Journal de la Societe francaise de statistique, 155(2),
    57-71.

.. [7] França, G., Rizzo, M. L., & Vogelstein, J. T. (2020). `Kernel k-Groups via Hartigan’s Method
    <https://ieeexplore.ieee.org/abstract/document/9103121>`_. IEEE transactions on pattern analysis and machine
    intelligence, 43(12), 4411-4425.

.. [8] Lemhadri, I., Ruan, F., Abraham, L., & Tibshirani, R. (2021). `LassoNet: A Neural Network with Feature Sparsity
    <https://lassonet.ml/>`_. Journal of Machine Learning Research, 22(127), 1–29.

.. [9] Ohl, L., Mattei, P. A., Bouveyron, C., Harchaoui, W., Leclercq, M., Droit, A., & Precioso, F. (2022).
    `Generalised Mutual Information for Discriminative Clustering
    <https://proceedings.neurips.cc/paper_files/paper/2022/hash/16294049ed8de15830ac0b569b97f74a-Abstract-Conference.html>`_.
    Advances in Neural Information Processing Systems, 35, 3377-3390.

.. [10] Ohl, L., Mattei, P. A., Bouveyron, C., Leclercq, M., Droit, A., & Precioso, F. (2024).
    `Sparse and Geometry-Aware Generalisation of the Mutual Information for Joint Discriminative Clustering and Feature
    Selection <https://link.springer.com/article/10.1007/s11222-024-10467-9>`_. Statistics and Computing, 34(5), 155.

.. [11] Ohl, L., Mattei, P. A., Leclercq, M., Droit, A., & Precioso, F. (2024). `Kernel KMeans Clustering Splits for
    End-to-End Unsupervised Decision Trees <https://arxiv.org/abs/2402.12232>`_. arXiv preprint arXiv:2402.12232.
