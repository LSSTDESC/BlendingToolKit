Installation
===============

*BlendingToolKit* (``btk``) uses `GalSim <https://github.com/GalSim-developers/GalSim>`_ to simulate galaxy images. The required python version for ``btk`` is ``python>=3.8``. 
The required packages for ``btk`` are:

* astropy
* galsim
* matplotlib
* numpy
* fitsio
* scikit-image 
* scipy
* sep

And their specific versions are listed in the ``pyproject.toml`` file under the ``[tool.poetry.dependencies]`` block.

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

    pip install blending_toolkit

This should install all other missing dependencies if necessary. You can then import the package as follows:

.. code-block:: python

    import btk
    import btk.catalog
