Installation
===============

*BlendingToolKit* (``btk``) uses `GalSim <https://github.com/GalSim-developers/GalSim>`_ to simulate galaxy images. The required python version for ``btk`` is ``python 3.9``.
The required packages for ``btk`` are:

* astropy
* galsim
* surveycodex
* matplotlib
* numpy
* scikit-image
* scipy
* sep
* tqdm

and their specific versions are listed in the `pyproject.toml <https://github.com/LSSTDESC/BlendingToolKit/blob/main/pyproject.toml>`_ under the ``[tool.poetry.dependencies]`` section.

Install GalSim
-------------------------------

GalSim is a python module that has much of its implementation in C++ for
improved computational efficiency. It can be installed with:

.. code-block::

    pip install galsim

However, you may have to install FFTW and Eigen manually. Refer to
`this <https://github.com/GalSim-developers/GalSim/blob/releases/2.1/INSTALL.md>`_
for more details.


Install BTK
------------------------------
Once you installed ``galsim``, you can install the latest released version of ``btk`` with ``pip``:

.. code-block::

    pip install --pre blending-toolkit

The version flag is necessary as the latest version is a pre-release. This command should install all other missing dependencies if necessary. You can then import the package as follows:

.. code-block:: python

    import btk
    import btk.catalog

Scarlet
------------------------------
BTK includes the Scarlet deblender as one of its ``Deblender`` classes. This means that you can easily run the scarlet deblender on BTK blends.

First you need to install scarlet, this is not by default installed with BTK as scarlet
is not in pypi. You can install scarlet by following the instructions here: `<https://pmelchior.github.io/scarlet/install.html>`_.
