Installation
===============

*BlendingToolKit* (btk) is essentially a wrapper around the
`WealLensingDeblending <https://weaklensingdeblending.readthedocs.io/en/latest/>`_
which uses `GalSim <https://github.com/GalSim-developers/GalSim>`_ to simulate
the galaxy images. So these packages along with
their dependencies need to be installed first.

The following dependencies are pip installable:

* numpy
* astropy
* fitsio
* scipy
* lmfit

Install Galsim
-------------------------------

GalSim is a python module that has much of its implementation in C++ for
improved computational efficiency. It can be installed with
::
    pip install galsim

However you may have to install FFTW and Eigen manually. Refer
`this <https://github.com/GalSim-developers/GalSim/blob/releases/2.1/INSTALL.md>`_
for mre details.

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

then install using
::
    cd BlendingToolKit
    python setup.py install

Optional Packages
-------------------------------

*BlendingToolKit* is meant to perform detection/deblending/measurement with any
user input algorithm. Thus, as such no algorithm is hard coded into the basic
framework. However, the tutorial notebooks do include examples of how these can
be performed with btk.

The tutorial notebooks require:

#. scarlet_ (multi-band deblender)
#. sep_ (Python library for Source Extraction and Photometry)
#. lsst_ science pipeline


.. _scarlet: https://scarlet.readthedocs.io/en/latest/index.html
.. _sep: https://sep.readthedocs.io/en/v1.0.x/index.html
.. _numpy: http://www.numpy.org
.. _lsst: https://pipelines.lsst.io
