.. btk documentation master file, created by
   sphinx-quickstart on Tue Mar 19 15:40:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:tocdepth: 3

.. image:: images/btk.png
   :align: center
   :scale: 30 %

*BlendingToolKit*
===============================
*BlendingToolKit* (BTK) is a framework to generate images of blended objects and evaluate performance metrics for different detection, deblending and measurement algorithms.

Detecting and separating overlapping sources, or "deblending", is primarily a research problem with several potential algorithmic solutions, including machine learning approaches. Computation of performance metrics on identical datasets will enable comparison between different algorithms. The goals of the btk framework are to allow the user to easily and quickly generate datasets of blended objects for testing different detection, deblending and measurement algorithms, as well as training samples for machine learning algorithms.

BTK is still under development and its structure is susceptible to change in the future.

Getting Started
==================
.. toctree::
   :maxdepth: 1

   install
   catalog
   user_guide
   tutorials

Modules API Reference
---------------------

.. toctree::
   :maxdepth: 2

   src/btk
