Installation
===============

*BlendingToolKit* (btk) is essentially a wrapper around the
`WeakLensingDeblending <https://weaklensingdeblending.readthedocs.io/en/latest/>`_
package, which uses `GalSim <https://github.com/GalSim-developers/GalSim>`_ to simulate galaxy images.
These packages along with their dependencies need to be installed first.

The following dependencies are pip installable:

* numpy
* astropy
* fitsio
* scipy
* lmfit

Install fitsio and lmfit
-----------------------------
These two uncommon packages can be installed via:
::
    # in conda.
    conda install -c conda-forge fitsio
    conda install -c conda-forge lmfit

    # in pip.
    pip install fitsio
    pip install lmfit


Install GalSim
-------------------------------

GalSim is a python module that has much of its implementation in C++ for
improved computational efficiency. It can be installed with
::
    pip install galsim

However you may have to install FFTW and Eigen manually. Refer to
`this <https://github.com/GalSim-developers/GalSim/blob/releases/2.1/INSTALL.md>`_
for more details.

Install WeakLensingDeblending package
---------------------------------------

WeakLensingDeblending package must first be cloned:
::
    git clone https://github.com/DarkEnergyScienceCollaboration/WeakLensingDeblending.git

Then run the following inside the WeakLensingDeblending folder:
::
    cd WeakLensingDeblending
    python setup.py install

Install *BlendingToolKit* with GIT
------------------------------------

The code is hosted on `github <https://github.com/LSSTDESC/BlendingToolKit>`_.
First download the repo:
::
    git clone https://github.com/LSSTDESC/BlendingToolKit.git

Then install using
::
    cd BlendingToolKit
    python setup.py install

Optional Packages
-------------------------------

*BlendingToolKit* is meant to perform detection/deblending/measurement with any
user input algorithm; therefore, no detection/deblending/measurement algorithm is hard-coded into the basic
framework. However, the tutorial notebooks include several examples of detection/deblending/measurement
algorithms that can be performed with btk.
These tutorial notebooks require:

#. scarlet_ (multi-band deblender)
#. sep_ (Python library for Source Extraction and Photometry)
#. lsst_ (LSST science pipeline)


.. _scarlet: https://scarlet.readthedocs.io/en/latest/index.html
.. _sep: https://sep.readthedocs.io/en/v1.0.x/index.html
.. _numpy: http://www.numpy.org
.. _lsst: https://pipelines.lsst.io
