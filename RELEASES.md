# History of changes

## In development

+ Adding the dynamic version of paths for feature selection in sparse models. A simply argument `dynamic=True` activates the dynamic mode.
+ Possibility of passing custom kernels and metrics to sparse models. This is not compatible with the dynamic mode.
+ No need to specify any longer the full partition of the features in the `groups` arguments of the sparse models
+ New GEMINIs: `HellingerGEMINI`, `TVGEMINI` and `KLGEMINI`
+ Introducing generic models that can be combined with any GEMINI
  + `gemclus.linear.LinearModel`, `gemclus.mlp.MLPModel`, `gemclus.nonparametric.CategoricalModel`,
    `gemclus.sparse.SparseLinearModel`, `gemclus.sparse.SparseMLPModel`
  + The GEMINI parametrisation of DOUGLAS can now be done through string
  + The dedicated MMD and Wasserstein models remain and support custom kernel/metric parameters
+ Fusing GEMINIs into a single class per distance
  + `gemclus.gemini.MMDOvA` and `gemclus.gemini.MMDOvO` are now `gemclus.gemini.MMDGEMINI`
  + `gemclus.gemini.WassersteinOvA` and `gemclus.gemini.WassersteinOvO` are now `gemclus.WassersteinGEMINI`
  + Both the MMD and Wasserstein GEMINI now support custom kernel/metric parameters
+ Fixing a gradient mistake in the `gemclus.MI`

## 0.2.0 (Latest)

+ Adding a new sparse logistic regression model trained with mutual information instead of MMD GEMINI: `gemclus.sparse.SparseLinearMI`
+ Adding new package containing methods for unsupervised tree clustering with end-to-end training: `gemclus.tree`. The package features a CART-like algorithm for clustering named `Kauri` and a differentiable tree named `Douglas`
+ *Experimental*: A method for adding constraints of type must-link cannot-link to discriminative models: `gemclus.add_mlcl_constraint`
+ Minor fixes in documentation
+ Better compatibility with scikit learn 1.3.0 regarding parameter constraint check

## 0.1.1

+ Fixing the ABCMeta parameter validation problem for the `draw_gmm` method for retrocompatibility with Python 3.8.
+ Constraining the package to Python>=3.8 to respect the requirements of the package.
+ Minor fix on the `get_selection` method for the Linear sparse models to respect the 1d output shape of the array.

## 0.1.0

+ Isolating the definition of GEMINIs in a separate classes for external usages: `gemini.MMDOvO`, `gemini.WassersteinOvA` etc.
+ Adding the nonparametric models in package `gemclus.nonparametric` with 2 additional examples for its usage in graph node clustering.
+ Fixing control variables in the `path` method for spars models.

## 0.0.2

+ Adding the Gaussian+Student-t mixture dataset: `gstm`
+ Method for sampling multivariate student-t distributions: `multivariate_student_t`
+ Adding tests for `data` package
+ Adding the possibility of a precomputed kernel/distance passed to `fit`
+ Adding batch size parameters
+ Fixing zero division in sparse linear model proximal gradient