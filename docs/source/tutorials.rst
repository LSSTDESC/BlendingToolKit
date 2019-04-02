Tutorials
=================

The following jupyter notebooks are included in the `notebooks/` directory:

Run basic btk (*run_basic.ipynb*)
-----------------------------------

This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/%2315/notebooks/run_basic.ipynb>`_ shows the how btk can generate multi-band blend image scenes along with isolated object images. PSF convolved object images are drawn in isolation and in the blend for each band.

The notebook shows examples of performing:

* detection with `SEP <https://sep.readthedocs.io/en/v1.0.x/index.html>`_ , `lsst science pipeline <https://pipelines.lsst.io>`
* deblending with `scarlet <https://scarlet.readthedocs.io/en/latest/index.html>`_
* segmentation with SEP.

The notebook also includes functions helpful for plotting multi-band images.

Run with user input custom functions (*custom_sampling_function.ipynb*)
--------------------------------------------------------------------------

This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/%2315/notebooks/custom_sampling_function.ipynb>`_ demonstrates how users can define their own sampling functions to draw blend image scenes.

Also shown is how a user defined custom function can generate different observing conditions. This would enable on the fly generation of blend scene images with different noise levels and observing PSFs.


Evaluate Metrics with btk (*evaluate_metrics.ipynb*)
----------------------------------------------------
This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/%2315/notebooks/evaluate_metrics.ipynb>`_ shows how to test performance of different detection/deblending/measurement algorithms. At present this only assesses detection performance by returning the number of objects correctly detected, number undetected and number of spurious detections.

5. Draw blends (*create_blend_generator*): Simulates scene of overlapping objects, convolved by PSF and with pixel noise (option set in *config*). Scene image is generated in each observing band. Isolated image of each object is also drawn without pixel contributions from other objects, in each band.


