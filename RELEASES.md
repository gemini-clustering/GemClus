# History of changes

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