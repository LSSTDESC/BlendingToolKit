Installation
===============

*BlendingToolKit* (``btk``) uses `GalSim <https://github.com/GalSim-developers/GalSim>`_ to simulate galaxy images. The python version required for ``btk`` is ``python 3.10`` or higher.
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

and their specific versions are listed in the `pyproject.toml <https://github.com/LSSTDESC/BlendingToolKit/blob/main/pyproject.toml>`_ under the ``dependencies`` section.

Install GalSim
-------------------------------

GalSim is a python module that has much of its implementation in C++ for
improved computational efficiency. It can be installed with:

.. code-block::

    pip install galsim

However you may have to install FFTW and Eigen manually. Refer to `this link <https://galsim-developers.github.io/GalSim/_build/html/install_pip.html>`_
for more details.

Another option is to create a ``conda`` environment and install GalSim through ``conda``. A minimal working example (after having installed ``conda``) looks like:

.. code-block::

    conda create -n py313 python=3.13 anaconda
    conda activate py313
    conda install -c conda-forge galsim

For more details see `this link <https://galsim-developers.github.io/GalSim/_build/html/install_conda.html>`_.



Install BTK
------------------------------
Once you installed ``galsim``, you can install the latest version of ``btk`` with ``pip``:

.. code-block::

    pip install blending-toolkit

This command should install all other missing dependencies if necessary. You can then import the package as follows:

.. code-block:: python

    import btk
    import btk.catalog

Scarlet
------------------------------
BTK includes the Scarlet deblender as one of its ``Deblender`` classes. This means that you can easily run the Scarlet deblender on BTK blends.

To use this functionality you need to install the ``scarlet`` package, which is not installed by default as ``scarlet`` is not in PyPI. You can install scarlet by following the instructions here: `<https://pmelchior.github.io/scarlet/install.html>`_.
