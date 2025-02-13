#! /usr/bin/env python

import codecs
import os

import numpy as np
from setuptools import find_packages, setup
from Cython.Build import cythonize
from setuptools.extension import Extension

ROOT = os.path.abspath(os.path.dirname(__file__))

setup(zip_safe=False,  # the package can run out of an .egg file
      packages=find_packages(),
      ext_modules=cythonize(Extension(
            name="gemclus.tree._utils",
            sources=["gemclus/tree/_utils.pyx"],
            language="c++",
            include_dirs=[np.get_include(), os.path.join(ROOT, "gemclus/tree")]
        )),
      )
