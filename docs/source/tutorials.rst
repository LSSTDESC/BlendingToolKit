Tutorials
=================

The following jupyter notebooks are included in the `notebooks/` directory:

Run basic btk (*run_basic.ipynb*).
-----------------------------------

This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/run_basic.ipynb>`_ shows how btk can be used to generate images of multi-band blend scenes, along with isolated object images -- i.e., PSF-convolved object images are drawn both in isolation and in the blend scene for each band.

The notebook shows examples of performing:

* detection with `SEP <https://sep.readthedocs.io/en/v1.0.x/index.html>`_, `lsst science pipeline <https://pipelines.lsst.io>`_,
* deblending with `scarlet <https://scarlet.readthedocs.io/en/latest/index.html>`_,
* segmentation with SEP.

The notebook also includes functions helpful for plotting multi-band images.

Run with user-input custom functions (*custom_sampling_function.ipynb*).
--------------------------------------------------------------------------

This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/custom_sampling_function.ipynb>`_ demonstrates how users can define their own sampling functions to draw blend image scenes.

Also shown is an example of how a user-defined custom function can be used to generate different observing conditions. This enables on-the-fly generation of blend scene images with different noise levels and observing PSFs.


Evaluate detection metrics with btk (*detection_metrics.ipynb*).
------------------------------------------------------
This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/detection_metrics.ipynb>`_ shows how to test the performance of different detection algorithms. At present this only assesses detection performance by returning the number of objects correctly detected, the number of undetected objects and the number of spurious detections.

Draw blends (*create_blend_generator*): Simulates a scene of overlapping objects, convolved by a PSF, with pixel noise (option set in *config*). The scene image is generated in each observing band. An isolated image of each object is also drawn in each band, with no pixel contributions from other objects.


