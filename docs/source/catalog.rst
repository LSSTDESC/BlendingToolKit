Input Catalog
===============

*BlendingToolKit* (btk) requires an input catalog that contains information required to simulate galaxies and blends.


CatSim
-------
The catalog simulation framework (CatSim) is a database of astrophysical sources with properties that are representative of what the LSST will observe at its ten-year coadded depth. Refer to the official CatSim `page <https://www.lsst.org/scientists/simulations/catsim>`_ for more details.

BTK includes a sample input catalog that contains parameters of 100 galaxies. A more extensive catalog can be downloaded from `this page <https://stanford.box.com/s/s1nzjlinejpqandudjyykjejyxtgylbk>`_.

A valid catsim catalog should at least contain the following columns:

- `ra`: Right Ascension (degrees)
- `dec`: Declination (degrees)
- `fluxnorm_bulge`: Proportion of flux in the bulge (fraction)
- `fluxnorm_disk`: Proportion of flux in the disk (fraction)
- `fluxnorm_agn`: Proportion of flux in the agn (fraction)
- `a_b` : Bulge half-light semi-major axis  (arcsec)
- `a_d` : Disk half-light semi-major axis  (arcsec)
- `b_b` : Bulge half-light semi-minor axis  (arcsec)
- `b_d` : Disk half-light semi-minor axis  (arcsec)
- `theta_b` : Bulge position angle (degrees)
- `pa_disk` : Disk position angle (degrees)
- `pa_bulge` : Bulge position angle (degrees), should be equivalent to `pa_disk`.

Additionally, the catalog must contain columns corresponding to the ab magnitudes for each the filters in the survey you selected. For example, for the default LSST survey in galcheat we require `i_ab`, `r_ab`, `u_ab`, `y_ab`, `z_ab`, and `g_ab` as found `here <https://github.com/aboucaud/galcheat/blob/main/galcheat/data/LSST.yaml>`_. Please see the `tutorial <tutorials.html>`_ and `user guide <user_guide.html>`_ for more information about surveys.



COSMOS
-------
The Cosmic Evolution Survey (COSMOS) is an astronomical survey based on multiple telescopes which provides a large number of multi-wavelength galaxy images, both as real images and parametric models. Refer to the `GalSim page for COSMOS <https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data>`_ for more details on how to get COSMOS data and to the `official COSMOS website <https://cosmos.astro.caltech.edu/>`_ for more information on the COSMOS catalog in general.

BTK includes an implementation of COSMOS real galaxy rendering, providing the possibility to get more realistic blends.


.. Cosmo DC2
.. ---------
.. `CosmoDC2 <https://arxiv.org/abs/1907.06530>`_ is a large synthetic galaxy catalog designed to support precision dark energy science with the Large Synoptic Survey Telescope (LSST). Refer to this `notebook <https://github.com/LSSTDESC/WeakLensingDeblending/blob/cosmoDC2_ingestion/notebooks/wld_ingestion_cosmoDC2.ipynb>`_ on how to inject the DC2 catalog into a CatSim-like catalog that can be analyzed with btk. The btk package includes a sample input catalog that contains parameters of 15 blend scenes with 61 galaxies.
