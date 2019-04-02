User Guide
==============

This document gives a detailed look into the work-flow of *BlendingToolKit* (btk). For a quick tutorial on how to run btk, see the JuPyter notebook tutorials :doc:`here </tutorials>`. This page is especially useful if you plan on using btk with your own detection/deblending/measurement algorithm.


The work-flow of btk is shown below

.. image:: images/flow_chart.png
   :align: center


1. Set parameters (*config*): define parameters to create postage stamps, including size of stamp, number of objects per blend, and how many stamps are to be drawn in one batch, i.e yielded at once by btk. The observing survey name (e.g. LSST. DES) is set here along with name of the input catalog and names of observing bands.
2. Load Catalog (*load_catalog*): Reads the input catalog file. This step includes an option to input a selection criteria.
3. Make blend catalog (*create_blend_generator*): Samples objects from the input catalog based on a user defined sampling function to create a catalog with parameters of each blend.  This step output a generator that yields a new set of catalogs each time run with *next()*. This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/%2315/notebooks/custom_sampling_function.ipynb>`_ shows different examples of user input sampling functions to create different blend catalogs.
4. Make observing conditions (*create_observing_generator*): Creates a class that contains information about the observing conditions like PSF, noise level, etc. The default is to use the full depth values corresponding to the survey named in config. However a user can define a function to generate this. This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/%2315/notebooks/custom_sampling_function.ipynb>`_ shows an example of a user input function to generate observing conditions.
5. Draw blends (*create_blend_generator*): Simulates scene of overlapping objects, convolved by PSF and with pixel noise (option set in *config*). Scene image is generated in each observing band. Isolated image of each object is also drawn without pixel contributions from other objects, in each band.
6. Detection/Deblending/Measure (*measure*): A user defined class to perform detection/deblending/measurement goes here. *btk* does not include any algorithm as default but only provides a a framework for the user to run their algorithm for the images generated in *create_blend_generator*. This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/%2315/notebooks/run_basic.ipynb>`_ contains examples of running btk with `SEP <https://sep.readthedocs.io/en/v1.0.x/index.html>`_ (SExtractor with python) , `lsst science pipeline <https://pipelines.lsst.io>`_ and `scarlet <https://scarlet.readthedocs.io/en/latest/index.html>`_.
7. Compute metrics (*metrics*): compares the true centroids, shapes, flux values to those predicted by the user algorithm. At present this only assesses detection performance by returning the number of objects correctly detected, number undetected and number of spurious detections. This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/%2315/notebooks/evaluate_metrics.ipynb>`_ shows how this can be done.


Utils
-------
*btk/utils.py* contains functions that the user may find useful in creating functions to perform detection/deblending/measurement in *measure*. It shows how a class derived from *measure.Measurement_params* can be defined by the user for use with SEP, lsst science pipeline or stand-alone scarlet.
