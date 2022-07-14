Using COSMOS galaxies
======================

In this section we will demonstrate how to generate blends using galaxies from the COSMOS catalog. You will find that generating images with COSMOS is very similar to generating images with Catsim.

Let's start with the catalog and sampling function. We use a small sample of the real COSMOS catalog that is already in the BTK repository, but you can fill in a different path if you have the complete data set on your computer. It can be downloaded from `at this page <https://zenodo.org/record/3242143>`_.

.. code:: ipython3

  COSMOS_CATALOG_PATHS = [
      "../data/cosmos/real_galaxy_catalog_23.5_example.fits",
      "../data/cosmos/real_galaxy_catalog_23.5_example_fits.fits",
  ]
  stamp_size = 24.0
  batch_size = 8
  catalog = btk.catalog.CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
  sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)

We can now create the corresponding instance of ``DrawBlendsGenerator``. There is an important caveat here: as in the other tutorial, we use the LSST survey. However, the COSMOS data set only contains images and magnitudes from the f814w band; thus, when simulating images, the same magnitude is used to compute the galaxy fluxes across all bands. The section that follows explains how to get around this issue.

.. code:: ipython3

  draw_generator = btk.draw_blends.CosmosGenerator(
          catalog,
          sampling_function,
          btk.survey.get_surveys("LSST"),
          batch_size=batch_size,
          stamp_size=stamp_size,
          cpus=1,
          add_noise="all",
          verbose=False,
      )

.. code:: ipython3

  batch = next(draw_generator)
  blend_images = batch['blend_images']
  blend_list = batch['blend_list']
  btk.plot_utils.plot_blends(blend_images, blend_list, limits=(30,90))


Using different magnitudes for each band
''''''''''''''''''''''''''''''''''''''''''''

In order to circumvent the aforementioned caveat, BTK offers the possibility to retrieve different magnitudes for each band. In order to use this feature, the corresponding magnitudes can be specified in any of the two provided COSMOS catalogs using the following column name format: ``"sn_fn"``, where ``sn`` and ``fn`` are the Survey and Filter names, respectively, as written in the ``Survey`` and ``Filter`` named tuple classes. BTK will automatically look for those columns and use the information when available to compute galaxy fluxes.

More information about the COSMOS catalog
''''''''''''''''''''''''''''''''''''''''''''

To better understand how to provided custom COSMOS data to BTK, let's review in more detail the COSMOS dataset and its implementation in BTK.

As seen :ref:`above <Using COSMOS galaxies>`, the BTK ``CosmosCatalog`` is instantiated from two COSMOS catalog files. The first one contains all the necessary information to draw a real galaxy (such as the paths to the galaxy and PSF stamps or the noise characteristics). The second one contains information about parameters fits to the galaxies (such as sersic parameters or bulge-to-disk ratios). You can refer to the galsim `documentation <https://galsim-developers.github.io/GalSim/_build/html/real_gal.html>`_ for more details. You can refer to the `COSMOS_23.5_training_sample_readme.txt` and `COSMOS_25.2_training_sample_readme.txt` README files coming with the COSMOS data set `download <https://zenodo.org/record/3242143>`_ to check the column details of each catalog.

In BTK, both the 'parametric' and 'real' mode to draw galaxies can be used. When drawing 'real' galaxies, most of the information of the second catalog is not necessary, but the file must be provided to instantiate the ``CosmosCatalog`` and ``galsim.COSMOSCatalog`` objects. In practice, BTK uses the ``flux_radius`` column to compute an estimate of the size of each source used for performance evaluation measures, so the second catalog should contain at least this column.

Custom COSMOS catalogs to draw 'real' galaxies should thus satisfy the following conditions:

1. The second catalog should contain at least the ``flux_radius`` column,

2. The first catalog should contain the same columns than the official COSMOS data release

3. The galaxy and PSF stamps should be provided and accessible.

4. (optional) One of the two catalogs can contain multiband magnitudes using the format described :ref:`above <Using different magnitudes for each band>`.

SCARLET implementation
=======================

We provide an implementation of the measure function for `SCARLET <https://www.sciencedirect.com/science/article/abs/pii/S2213133718300301>`_ , a deblending algorithm based on matrix factorization. The code for SCARLET can be found in this `repo <https://github.com/pmelchior/scarlet>`_. You can install scarlet and its dependencies directly along BTK by running

::

  pip install btk[scarlet]
  pip install git+https://github.com/pmelchior/scarlet

This will install the latest version of SCARLET in github and NOT in pip (which is outdated).

You can find the SCARLET measure function implementation `here <https://github.com/LSSTDESC/BlendingToolKit/blob/main/notebooks/01b-scarlet-measure.ipynb>`_.

Advanced features
==================

You can find more details on specific features of BTK in these two tutorials: `the first one <https://github.com/LSSTDESC/BlendingToolKit/blob/main/notebooks/02b-custom-tutorial.ipynb>`_ explains how to write your own sampling function, survey or measure function (the measure function may be particularily important for users who want to test their own algorithm. `The second one <https://github.com/LSSTDESC/BlendingToolKit/blob/main/notebooks/02a-multi-tutorial.ipynb>`_ details how to use the multiresolution feature, as well as how to deal with multiple measure functions and how to pass them several different arguments using the "measure_kwargs".
