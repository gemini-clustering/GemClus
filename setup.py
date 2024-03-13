#! /usr/bin/env python

import codecs
import os
import re

import numpy as np
from setuptools import find_packages, setup
from Cython.Build import cythonize
from setuptools.extension import Extension

# get __version__ from _version.py
ver_file = os.path.join('gemclus', '__init__.py')
with open(ver_file) as f:
    __version__ = re.search(
        r"__version__\s*=\s*['\"]([^'\"]*)['\"]",
        f.read()).group(1)

DISTNAME = 'gemclus'
DESCRIPTION = 'A package for performing discriminative clustering with gemini-trained models'
with codecs.open('README.md', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = 'Louis Ohl'
AUTHOR_EMAIL = 'louis.ohl@inria.fr'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
URL = 'https://github.com/gemini-clustering'
LICENSE = 'Dual license: GPLv3 and Commercial'
DOWNLOAD_URL = 'https://github.com/gemini-clustering'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'pot>=0.8.1']
SETUP_REQUIRES = ["numpy", "cython>=0.23"]
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Intended Audience :: Education',
               'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
               'Programming Language :: Python',
               'Topic :: Utilities',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Topic :: Scientific/Engineering :: Information Analysis',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               'Programming Language :: Python :: 3.10']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}
ROOT = os.path.abspath(os.path.dirname(__file__))

setup(name=DISTNAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      ext_modules=cythonize(Extension(
            name="gemclus.tree._utils",
            sources=["gemclus/tree/_utils.pyx"],
            language="c++",
            include_dirs=[np.get_include(), os.path.join(ROOT, "gemclus/tree")]
        )),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      setup_requires=SETUP_REQUIRES)
