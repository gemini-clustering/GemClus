from gemclus import sparse, linear, mlp, data, nonparametric, gemini, tree
from ._base_gemini import _BaseMMD, _BaseWasserstein
from .mlcl import add_mlcl_constraint

# Defining aliases for the clarity of outer codes
WassersteinModel = _BaseWasserstein
MMDModel = _BaseMMD

__all__ = ['linear', 'mlp', 'sparse', 'data', 'nonparametric', 'gemini', 'tree', 'add_mlcl_constraint', 'MMDModel',
           'WassersteinModel', '__version__']

__version__ = '0.2.0'
