:tocdepth: 3

.. image:: images/btk.png
   :align: center
   :scale: 30 %

*BlendingToolKit*
===============================
*BlendingToolKit* (BTK) is a framework to generate images of blended objects and evaluate performance metrics on various detection, deblending and measurement algorithms.

Detecting and separating overlapping sources, or "deblending", is primarily a research problem with several potential algorithmic solutions, including machine learning approaches. Computation of performance metrics on identical datasets will enable comparison between different algorithms. The goals of the BTK framework are to allow the user to easily and quickly generate datasets of blended objects for testing different detection, deblending and measurement algorithms, as well as training samples for machine learning algorithms.

If you use this software for your research, please use the following bibtex entry to cite:

::

   @ARTICLE{mendoza2024btk,
         author = {{Mendoza}, Ismael and {Torchylo}, Andrii and {Sainrat}, Thomas and {Guinot}, Axel and {Boucaud}, Alexandre and {Paillassa}, Maxime and {Avestruz}, Camille and {Adari}, Prakruth and {Aubourg}, Eric and {Biswas}, Biswajit and {Buchanan}, James and {Burchat}, Patricia and {Doux}, Cyrille and {Joseph}, Remy and {Kamath}, Sowmya and {Malz}, Alex I. and {Merz}, Grant and {Miyatake}, Hironao and {Roucelle}, C{\'e}cile and {Zhang}, Tianqing and {the LSST Dark Energy Science Collaboration}},
         title = "{The Blending ToolKit: A simulation framework for evaluation of galaxy detection and deblending}",
         journal = {arXiv e-prints},
      keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics},
            year = 2024,
         month = sep,
            eid = {arXiv:2409.06986},
         pages = {arXiv:2409.06986},
            doi = {10.48550/arXiv.2409.06986},
   archivePrefix = {arXiv},
         eprint = {2409.06986},
   primaryClass = {astro-ph.IM},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240906986M},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }


Getting Started
==================
.. toctree::
   :maxdepth: 1

   install
   user_guide
   catalog
   tutorials

Modules API Reference
---------------------

.. toctree::
   :maxdepth: 3

   src/btk
