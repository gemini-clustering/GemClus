#################################
Welcome to GemClus documentation!
#################################

Welcome and thank you for checking GemClus out, this really pleasures us.

About the package
=================

What is GemClus?
----------------

GemClus is a Python package intended for discriminative clustering. This packages aims at providing different
clustering models that share the same discriminative nature, specifically in the sense of Minka [1]_.

Why GemClus?
------------

GemClus originates from our work on the *generalised mutual information* (GEMINI).
GEMINI is a clustering-dedicated function derived from information theory that allows to do clustering without hypotheses
on the data distributions. This work led us to realise that multiple discriminative models lacked implementations in
Python. We tried to bridge this gap by providing a tool that simultaneously offers all of the GEMINI spectrum and
implementations of other discriminative clustering methods. These methods include small neural networks, logistic
regression, decision trees and work from other paper that we will relevant to the discriminative clustering field in
the GEMINI spirit.

Scope of GemClus
----------------

The scope of this package is especially for small-scale models: we provide implementations of linear models, trees,
small neural networks using only NumPy. We try to provide also some synthetic datasets which could be of interest to
the scientific community. We welcome any novel contribution, missing discriminative model or even unimplemented dataset.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   api
   history

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

Contents
========

* `Getting started <quick_start.html>`_.
* `User Guide <user_guide.html>`_.
* `API <api.html>`_.
* `Examples <auto_examples/index.html>`_

.. include:: ../README.md
    :parser: myst_parser.sphinx_

References
==========

.. [1] Minka, T. (2005). `Discriminative models, not discriminative training
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2005-144.pdf>`_.
    Technical Report MSR-TR-2005-144, Microsoft Research.

.. image:: _static/images/logo_3ia.png
    :height: 80
    :alt: 3IA, Université Côte d'Azur

.. image:: _static/images/logo_ul.png
    :height: 80
    :alt: Université Laval

.. image:: _static/images/logo_inria.png
    :height: 80
    :alt: INRIA

.. image:: _static/images/logo_i3s.png
    :height: 80
    :alt: Laboratoire d Informatique Signaux et Systèmes
