#! /usr/bin/env python

import codecs
import os
import re

from setuptools import find_packages, setup

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
MAINTAINER = 'Louis Ohl'
MAINTAINER_EMAIL = 'louis.ohl@inria.fr'
URL = 'https://github.com/gemini-clustering'
LICENSE = 'GPLv3'
DOWNLOAD_URL = 'https://github.com/gemini-clustering'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'pot>=0.8.1']
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
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
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

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
