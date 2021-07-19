Input Catalog
===============

*BlendingToolKit* (btk) requires an input catalog that contains information required to simulate galaxies and blends. Each galaxy is parameterized as a bulge + disk  + agn with parameters following the `LSST DM catalog schema <https://confluence.lsstcorp.org/display/SIM/Database+Schema>`_


CatSim
-------
The catalog simulation framework (CatSim) is a database of astrophysical sources with properties that are representative of what the LSST will observe at its ten-year coadded depth. Refer to the official CatSim `page <https://www.lsst.org/scientists/simulations/catsim>`_ for more details.
The btk package includes a sample input catalog that contains parameters of 100 galaxies. A more extensive catalog can be downloaded from `here <https://stanford.box.com/s/s1nzjlinejpqandudjyykjejyxtgylbk>`_.

COSMOS
-------
The Cosmic Evolution Survey (COSMOS) is an astronomical survey based on multiple telescopes which provides a large number of multi-wavelength galaxy images, both as real images and parametric models. Refer to the `GalSim page for COSMOS <https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data>`_ for more details on how to get COSMOS data and to the `official COSMOS website <https://cosmos.astro.caltech.edu/>`_ for more information on the COSMOS catalog in general.
BTK includes an implementation of COSMOS real galaxy rendering, providing the possibility to get more realistic blends.

Galsim_Hub
-----------
Galsim_Hub is a framework for deep learning image generation models, developped by François Lanusse, which is available `here <https://github.com/McWilliamsCenter/galsim_hub>`_ . BTK supports drawing galaxies generated using any galsim_hub compatible model; the default one generates COSMOS-like galaxies and is described in `this paper <https://arxiv.org/abs/2008.03833>`_, but the BTK implementation should be compatible with any model.

.. Cosmo DC2
.. ---------
.. `CosmoDC2 <https://arxiv.org/abs/1907.06530>`_ is a large synthetic galaxy catalog designed to support precision dark energy science with the Large Synoptic Survey Telescope (LSST). Refer to this `notebook <https://github.com/LSSTDESC/WeakLensingDeblending/blob/cosmoDC2_ingestion/notebooks/wld_ingestion_cosmoDC2.ipynb>`_ on how to inject the DC2 catalog into a CatSim-like catalog that can be analyzed with btk. The btk package includes a sample input catalog that contains parameters of 15 blend scenes with 61 galaxies.
