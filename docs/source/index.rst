.. btk documentation master file, created by
   sphinx-quickstart on Tue Mar 19 15:40:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*BlendingToolKit*
===============================
*BlendingToolKit* is a framework to generate images of blended objects and
evaluate performance metrics for different algorithms.

Detecting and separating overlapping sources, or "deblending", is primarily a
research problem with several potential algorithmic solutions, including machine
learning approaches. Thus it will be convenient to have a toolkit that can
easily and quickly generate datasets of blended objects for testing different
algorithms, along with training samples for machine learning algorithms.
Computation of performance metrics on identical datasets will enable comparison
between different algorithms. With these goals in mind, the development of
BlendingToolKit began.

Key features of the framework are:
* It can generate training/validation/test sets for developing and
testing detection, deblending and measurement algorithms.
* It includes data augmentation and independent (but reproducible) noise
realizations.
* It is easily customizable with options to include user defind functions for
blend generation, obseerving condtions, etc,.
produces images on the fly.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   project-summary
   API Reference <api/btk>

Getting Started
==================

   install
   quickstart
   user_docs
   tutorials
   diagnostics
   api_docs

Modules API Reference
---------------------

.. toctree::
   :maxdepth: 3

   src/btk
