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

Run btk with user-input custom functions (*custom_sampling_function.ipynb*).
----------------------------------------------------------------------------
btk generates postage stamp images of blend scenes from an input catalog. How a particular scene is sampled from this catalog is defined in a function input to *btk.create_blend_generator.generate*.
This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/custom_sampling_function.ipynb>`_ demonstrates how users can define their own sampling function to draw blend image scenes.

Also shown is an example of how a user-defined custom function can be used to generate different observing conditions. This enables on-the-fly generation of blend scene images with different noise levels and observing PSFs.


Evaluate detection metrics with btk (*detection_metrics.ipynb*).
----------------------------------------------------------------
This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/detection_metrics.ipynb>`_ shows how to test the performance of different detection algorithms for a test set size of 100 blend scenes. Detected centroids are compared to true centers to ascertain if a source was detected or not. A detection  efficiency matrix is computed that shows the fraction of detected objects for different number of objects in the blend scene. Also plotted are histograms of fraction of objects that were detected as a function of intrinsic galaxy parameters.


Run btk with an input config file (*with_config_file_input.ipynb*).
-------------------------------------------------------------------
This `notebook <https://github.com/LSSTDESC/BlendingToolKit/blob/master/notebooks/with_config_file_input.ipynb>`_ shows how to run btk with an input config yaml file. The input yaml config file contains information on how to simulate the blend scene, which detection/deblending/measurement algorithm to run and where to save the outputs. The config file is parsed by *btk_input.py* which then runs btk with the defined parameters.

The tutorial shows how to run *btk_input.py* with an example config file (*input/example-config.yaml*) for three types of simulations:

1. Two-galaxy blends sampled randomly from CatSim galaxies
2. Up to 10 galaxy blends sampled randomly from CatSim galaxies
3. Blends defined as galaxy "groups" from a pre-processed wld output

