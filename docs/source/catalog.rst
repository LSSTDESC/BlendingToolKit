Galaxy Catalogs
===============

*BlendingToolKit* (btk) requires an galaxy catalog that contains information required to simulate galaxies and blends.


CatSim
-------
The catalog simulation framework (CatSim) is a database of astrophysical sources with properties that are representative of what the LSST will observe at its ten-year coadded depth. Refer to the official CatSim `page <https://www.lsst.org/scientists/simulations/catsim>`_ for more details.

BTK includes a sample input catalog that contains parameters of approximately ``85k`` galaxies. A more extensive catalog can be downloaded from `this page <https://stanford.box.com/s/s1nzjlinejpqandudjyykjejyxtgylbk>`_.

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

Additionally, the catalog must contain columns corresponding to the ab magnitudes for each the filters in the survey you selected. For example, for the default LSST survey in galcheat we require `i_ab`, `r_ab`, `u_ab`, `y_ab`, `z_ab`, and `g_ab` as found `here <https://github.com/aboucaud/galcheat/blob/main/galcheat/data/LSST.yaml>`_.


COSMOS
-------
The Cosmic Evolution Survey (COSMOS) is an astronomical survey based on multiple telescopes which provides a large number of multi-wavelength galaxy images, both as real images and parametric models. Refer to the `GalSim page for COSMOS <https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data>`_ for more details on how to get COSMOS data and to the `official COSMOS website <https://cosmos.astro.caltech.edu/>`_ for more information on the COSMOS catalog in general.

BTK includes an implementation of COSMOS real galaxy rendering. This enables the possibility
to have use galaxies with more realistic morphology in the simulated blends.


Using different magnitudes for each band
''''''''''''''''''''''''''''''''''''''''''''

BTK offers the possibility to retrieve different magnitudes for each band. In order to use this feature, the corresponding magnitudes can be specified in any of the two provided COSMOS catalogs using the following column name format: ``"sn_fn"``, where ``sn`` and ``fn`` are the Survey and Filter names, respectively, as written in the ``Survey`` and ``Filter`` classes.
BTK will automatically look for those columns and use the information when available to compute galaxy fluxes.

More information about the COSMOS catalog
''''''''''''''''''''''''''''''''''''''''''''

To better understand how to provided custom COSMOS data to BTK, let's review in more detail the COSMOS dataset and its implementation in BTK.

The BTK ``CosmosCatalog`` is instantiated from two COSMOS catalog files. The first one contains all the necessary information to draw a real galaxy (such as the paths to the galaxy and PSF stamps or the noise characteristics). The second one contains information about parameters fits to the galaxies (such as sersic parameters or bulge-to-disk ratios). You can refer to the galsim `documentation <https://galsim-developers.github.io/GalSim/_build/html/real_gal.html>`_ for more details. You can refer to the `COSMOS_23.5_training_sample_readme.txt` and `COSMOS_25.2_training_sample_readme.txt` README files coming with the COSMOS data set `download <https://zenodo.org/record/3242143>`_ to check the column details of each catalog.

In BTK, both the 'parametric' and 'real' mode to draw galaxies can be used. When drawing 'real' galaxies, most of the information of the second catalog is not necessary, but the file must be provided to instantiate the ``CosmosCatalog`` and ``galsim.COSMOSCatalog`` objects. In practice, BTK uses the ``flux_radius`` column to compute an estimate of the size of each source used for performance evaluation measures, so the second catalog should contain at least this column.

Custom COSMOS catalogs to draw 'real' galaxies should thus satisfy the following conditions:

1. The second catalog should contain at least the ``flux_radius`` column,

2. The first catalog should contain the same columns than the official COSMOS data release

3. The galaxy and PSF stamps should be provided and accessible.

4. (optional) One of the two catalogs can contain multiband magnitudes using the format described :ref:`above <Using different magnitudes for each band>`.


.. Cosmo DC2
.. ---------
.. `CosmoDC2 <https://arxiv.org/abs/1907.06530>`_ is a large synthetic galaxy catalog designed to support precision dark energy science with the Large Synoptic Survey Telescope (LSST). Refer to this `notebook <https://github.com/LSSTDESC/WeakLensingDeblending/blob/cosmoDC2_ingestion/notebooks/wld_ingestion_cosmoDC2.ipynb>`_ on how to inject the DC2 catalog into a CatSim-like catalog that can be analyzed with btk. The btk package includes a sample input catalog that contains parameters of 15 blend scenes with 61 galaxies.
