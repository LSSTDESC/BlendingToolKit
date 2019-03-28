
:tocdepth: 1


  Installation
============
*BlendingToolKit* is essentially a wrapper around the WealLensingDeblending
which uses GalSim to simulate the galaxy images. So these packages along with
their dependencies need to be installed first.

The folllowing dependencies are pip installable:

*numpy
*astropy
*fitsio
*scipy
*lmfit


Install Galsim
----------------
GalSim is a python module that has much of its implementation in C++ for
improved computational efficiency. It can be installed with
::
    pip install galsim
However you may have to install FFTW and Eigen maunally. Refer
`this <https://github.com/GalSim-developers/GalSim/blob/releases/2.1/INSTALL.md>`
for mre details.


Install WeakLensingDeblending package
----------------

::
    git clone https://github.com/DarkEnergyScienceCollaboration/WeakLensingDeblending.git
and then run the following inside the WeakLensingDeblending folder
::
    python setup.py install

Install *BlendingToolKit* with GIT
----------------

The code is hosted on `github <https://github.com/LSSTDESC/BlendingToolKit>`.
First download the repo
::

    git clone https://github.com/LSSTDESC/BlendingToolKit.git
then istall using
::

    python setup.py install


