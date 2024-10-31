#####################################
Quick start on GemClus
#####################################

In GemClus, we distinguish models and their objective function. It is possible to pick any pair of model and objective
function to do clustering. Let us use for example a logistic regression or a multi-layered perceptron (MLP) as
clustering models. In GemClus, generic models come by default with the one-vs-all MMD GEMINI.

.. code-block:: python

    # Import model definitions
    from gemclus import mlp
    from gemclus import linear

    # Create a 2-layer neural network for clustering
    model = mlp.MLPModel()
    model.fit(X)
    p_y_given_x = model.predict_proba(X)

    # Create an unsupervised logistic regression
    model = linear.LinearModel()
    model.fit(X)
    p_y_given_x = model.predict_proba(X)

It is possible to select different GEMINI by using the `gemini` parameter. All available GEMINIs are listed in
`gemclus.gemini.AVAILABLE_GEMINIS`. For example, the one-vs-one total variation GEMINI can be accessed using the code
"tv_ovo".

.. code-block:: python

    model = mlp.MLPModel(gemini="tv_ovo")

Another approach is to instanciate the desired gemini and pass it as argument. Here, we pass the argument `ovo=True`.
Otherwise, geminis are in OvA mode by default.

.. code-block:: python

    from gemclus import gemini

    objective = gemini.TVGEMINI(ovo=True)
    model = mlp.MLPModel(gemini=objective)

The MMD and Wasserstein GEMINIs are a bit more special because they required a kernel (resp. a distance). It is
possible to use any kernel/distance in scikit-learn.

.. code-block:: python

    # Create OvA Wasserstein GEMINI with manhattan distance
    objective = gemini.WassersteinGEMINI(metric="manhattan")

It is further possible to pass parameters to this metric.  For example, if we use an RBF kernel in the MMD GEMINI,
we would like to set the scale parameter, named
`gamma <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html>`_ in scikit-learn.

.. code-block:: python

    # Set gamma parameter to 2 in rbf kernel
    objective = gemini.MMDGEMINI(kernel="rbf", kernel_params={"gamma":2})

If a kernel or a distance cannot among those proposed by default, it can be passed as an argument to the `fit` method.
To that end, the value "precomputed" must be passed to the `kernel` or `metric` argument.

.. code-block:: python

    # We use a pre-computed kernel in the objective
    objective = gemini.MMDGEMINI(kernel="precomputed")
    # So we pass the custom kernel inside the fit method
    model = linear.LinearModel(gemini=objective)
    model.fit(X, precomputed_kernel)

To simplify the above code when using MMD or Wasserstein distance in GEMINI, we propose models that directly incorporate
those GEMINIs in their constructor.  These specific models have the word `Model` replaced by `MMD`or `Wasserstein` in
their name, e.g. :class:`gemini.linear.LinearMMD` or :class:`gemini.mlp.MLPWasserstein`.

.. code-block:: python

    # Define a logistic regression trained by OvO MMD GEMINI
    model = linear.LinearMMD(ovo=True, kernel="rbf", kernel_params={"gamma":2})
    # Define a mlp trained with OvA Wasserstein GEMINI on a custom metric
    model = mlp.MLPWasserstein(metric="precomputed")


Discriminative models can be easily incorporated in the GemClus framework using inheritance from the base model
:class:`gemclus.base.DiscriminativeModel`. An example is given `here <auto_examples/_general/plot_custom_model.html>`_.