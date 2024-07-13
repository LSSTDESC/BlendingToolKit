User Guide
==============

This document gives a detailed look into the workflow of *BlendingToolKit* (btk). For tutorials on running BTK, take a look at our `tutorials <https://lsstdesc.org/BlendingToolKit/tutorials.html>`_. This page is especially useful if you plan to use BTK with your own detection or deblending algorithm.

The workflow is as follows:

.. image:: images/diagram.png
   :align: center

1. *Set parameter values*: define parameter values to create postage stamps, including size of stamp, number of objects per blend, and how many stamps are to be drawn in one batch -- i.e., the number of stamps btk will produce together in a singe batch. The observing survey name (e.g., LSST, DES), the name of the input catalog to draw objects from, and the names of the observing bands are also specified here.

2. *Load Catalog:* Reads the input catalog file. This must be done using a :class:`~btk.catalog.Catalog`-like object (e.g. :class:`~btk.catalog.CatsimCatalog`), either by providing directly the catalog or by using the :meth:`~btk.catalog.Catalog.from_file` method.

3. *Specify the* :class:`~btk.sampling_functions.SamplingFunction`: A sampling function is a callable object, which takes into input a catalog and returns specific entries along with parameters for a given blend (such as object shifts in the stamp for instance).
You may use the default class :class:`~btk.sampling_functions.DefaultSamplingFunction`, or define a new one if you want to have more control over how the galaxies are selected and the blends are defined. Please take a look at our `tutorials <https://lsstdesc.org/BlendingToolKit/tutorials.html>`_ page for more details.

4. *Choose a survey:* BTK now relies on the `surveycodex <https://github.com/LSSTDESC/surveycodex>`_ package. This package contains informations on various survey, including LSST, HSC and HST COSMOS (among others). It provides this information as a :class:`~surveycodex.survey.Survey` object, which can be easily imported in BTK using the survey name, via the `get_surveys` function. The user can also provide a custom PSF at this point, either as a `galsim` model or with a FITS file, or use the default (optical + atmospheric) PSF  provided by BTK.

5. *Draw blends*: Simulates scene of overlapping objects, convolved by the PSF and with pixel noise. This step is done by creating a :class:`~btk.draw_blends.DrawBlendsGenerator`-like object (e.g. :class:`~btk.draw_blends.CatsimGenerator`), which takes as input the catalog, sampling function and survey created in step 2 to 4. It can then be called using ``next(draw_blends_generator)`` to get the results as a `:class:btk.blend_batch.BlendBatch`, which includes attributes containing the blend images with the key ``blend_images``, the isolated galaxy images ``isolated_images``, and the blend parameters with the key ``catalog_list``.
Fluxes in BTK are calculated using `surveycodex` based on the :func:`~surveycodex.utilities.mag2counts` function and corresponding survey and filter parameters. Please see the `surveycodex <https://lsstdesc.org/surveycodex/api/utilities.html>` documentation for more details.

6. *Detection and Deblending*: Performs user-defined measurements (detection, segmentation, deblended images) on the generated blends. The user may create a subclass of  :class:`~btk.deblend.Deblender` to deblend the corresponding blends. We currently have 3 available 'deblenders':

- SourceExtractor in Python (SEP)
- Scarlet
- Peak-finding algorithm (from `scikit-image`)

 The user is also available to implement their own deblender and use it as part of BTK. See the `tutorials page <https://lsstdesc.org/BlendingToolKit/tutorials.html>`_ for details.

7. *Matching and Evaluation*: Matches the predicted detections to the true galaxies' centroids and compute metrics that evaluate the quality of the detection, segmentation and deblended images.
