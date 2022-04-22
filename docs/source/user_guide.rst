User Guide
==============

This document gives a detailed look into the workflow of *BlendingToolKit* (btk). For a quick tutorial on how to run btk, take a look at our `tutorials <tutorials.html>`_. This page is especially useful if you plan to use btk with your own detection/deblending/measurement algorithm. The workflow presented here should be as general as possible.

The workflow is as follows:

.. image:: images/current_flowchart.png
   :align: center


1. *Set parameter values*: define parameter values to create postage stamps, including size of stamp, number of objects per blend, and how many stamps are to be drawn in one batch -- i.e., the number of stamps btk will produce together in a singe batch. The observing survey name (e.g., LSST, DES), the name of the input catalog to draw objects from, and the names of the observing bands are also specified here.

2. *Load Catalog:* Reads the input catalog file. This must be done using a :class:`~btk.catalog.Catalog`-like object (e.g. :class:`~btk.catalog.WLDCatalog`), either by providing directly the catalog or by using the :meth:`~btk.catalog.Catalog.from_file` method.

3. *Specify the* :class:`~btk.sampling_functions.SamplingFunction`: A sampling function is a callable object, which takes into input a catalog and returns specific entries along with parameters for a given blend (such as object shifts in the stamp for instance). You may use the default class :class:`~btk.sampling_functions.DefaultSamplingFunction`, or define a new one if you want to have more control over how the galaxies are selected and the blends are defined. *NOTE:* The default sampling function performs a cut of ``ref_mag < 25.3``. Please take a look at our `tutorials <tutorials.html>`_ page for more details.

4. *Choose a survey:* BTK now relies on the `galcheat <https://github.com/aboucaud/galcheat>`_ package. This package contains informations on various survey, including LSST, HSC and HSC COSMOS (among others). It provides this information as a :class:`~galcheat.survey.Survey` object, which can be easily imported in BTK using its name, via the `get_surveys` function. It is also pretty simple to tweak the parameters of the survey at this stage. The user can also provide a custom PSF at this point, either as a Galsim model or with a FITS file (there is also a possibility to randomize the PSF if needed), or use the default PSF provided by BTK.

5. *Draw blends*: Simulates scene of overlapping objects, convolved by the PSF and with pixel noise. This step is done by creating a :class:`~btk.draw_blends.DrawBlendsGenerator`-like object (e.g. :class:`~btk.draw_blends.WLDGenerator`), which is given the catalog, sampling function and survey created in step 2 to 4. It can then be called using ``next(draw_blends_generator)`` to get the results as a dictionary, including the blends with the key ``blend_images``, the isolated galaxy images with the key ``isolated_images`` and the blend parameters with the key ``blend_list``. In the case where multiple surveys were provided in step 4, each entry will instead take the form of a dictionary indexed by the survey names, with each value corresponding to the information for one of the surveys. Fluxes in BTK are calculated using `galcheat` based on the :function:`~galcheat.utilities.mag2counts` function and corresponding survey and filter parameters. Please see the `galcheat` documentation for more details.

6. *Detection/Deblending/Measure*: Performs user-defined measurements (detection, segmentation, deblended images) on the generated blends. The user may create a :class:`~btk.measure.MeasureGenerator`, providing the draw blends generator from step 5 as well as one or several measure function(s), which perform the measurements on one blend. The user is expected to write his own measure function, an implementation for `SEP <https://sep.readthedocs.io/en/v1.0.x/index.html>`_ (SourceExtractor with python) is available as an example `here <https://github.com/LSSTDESC/BlendingToolKit/blob/ae833212127d5c5ec64a205f6731d9d1d03fdec0/btk/measure.py#L132>`_ and a more detailed explanation on how to implement your own measure function is available in the `tutorials page <tutorials.html>`_.

7. *Compute metrics*: Matches the detections to the true galaxies and compute metrics relative to the quality of the detection, segmentation and deblended images. This is achieved using a :class:`~btk.metrics.MetricsGenerator` object, which takes as an argument the measure generator from step 6. For users that do not wish to use the whole BTK framework, the function :func:`~btk.metrics.compute_metrics` can be used directly to compute the metrics by directly providing the data.
