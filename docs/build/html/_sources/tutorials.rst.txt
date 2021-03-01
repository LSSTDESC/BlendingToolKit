Tutorials
=============

NOTE : Most of the tutorials in this page are *outdated* but can still be found in the notebooks/broken directory, and might help to get an idea on how to use btk in certain cases. The most up-to-date tutorial as of now is this `introductory tutorial <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/intro.ipynb>`_ which should be able to get you started with btk.

The following jupyter notebooks are included in the `notebooks/` directory:

Run basic btk (*run_basic.ipynb*).
-----------------------------------

This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/run_basic.ipynb>`_ shows how btk can be used to generate images of multi-band blend scenes, along with isolated object images -- i.e., PSF-convolved object images are drawn both in isolation and in the blend scene for each band. The blend scenes are drawn with and without pixel noise.

The notebook also shows examples of performing:

* detection with `SEP <https://sep.readthedocs.io/en/v1.0.x/index.html>`_, `lsst science pipeline <https://pipelines.lsst.io>`_,
* deblending with `scarlet <https://scarlet.readthedocs.io/en/latest/index.html>`_,
* segmentation with SEP.

Multi-band images are plotted using functions defined in *btk.plot_utils*


Tests
------

Importing relevant modules

.. jupyter-execute::

  %matplotlib inline
  import matplotlib.pyplot as plt
  import numpy as np
  import os
  import sys
  import btk
  import btk.plot_utils
  import btk.survey
  import btk.draw_blends
  import astropy.table

Draw blends generator

.. jupyter-execute::

  stamp_size = 24.0 # arcsecs
  catalog_name = "../data/sample_input_catalog.fits"
  catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
  sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size, max_number=3)
  draw_generator = btk.draw_blends.CatsimGenerator(
      catalog,
      sampling_function,
      [btk.survey.Rubin],
      batch_size=8,
      stamp_size=stamp_size,
      shifts=None,
      indexes=None,
      multiprocessing=False,
      cpus=1,
      add_noise=True,
  )

Plot the blends

.. jupyter-execute::

  batch = next(draw_generator)
  blend_images = batch['blend_images']
  blend_list = batch['blend_list']
  btk.plot_utils.plot_blends(blend_images, blend_list, limits=(30,90))