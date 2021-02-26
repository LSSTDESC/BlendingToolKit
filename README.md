![tests](https://github.com/LSSTDESC/BlendingToolKit/workflows/tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/blendingtoolkit/badge/?version=latest)](https://blendingtoolkit.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/LSSTDESC/BlendingToolKit/branch/master/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/BlendingToolKit)

**NOTE:** BTK is currently undergoing heavy development and rapidly changing, as such the documentation and most jupyter could be deprecated. Please feel free to contact [@ismael-mendoza](https://github.com/ismael-mendoza) if you would like to use `BTK` for a current project or contribute.
# BlendingToolKit
Framework for fast generation and analysis of galaxy blends catalogs. This toolkit is a convenient way of
producing multi-band postage stamp images of blend scenes.

Documentation can be found at https://blendingtoolkit.readthedocs.io/en/latest/

## Workflow
<img src="docs/source/images/current_flowchart.png" alt="btk workflow" width="450"/>

Color code for this flowchart :
- Classes in black should be used as is by the user.
- Classes in red may be reimplemented by the experienced user ; we recommend for new users to use the default implementations until they are familiar with them.
- In blue is the code for instantiating the classes within the code (optionnal arguments not included).
- In green are the revelant methods for the classes ; please note that the `__call__` method is executed when calling the object (eg `sampling_function(catalog)`) and the `__next__` method is executed when using `next` (eg `next(generator)`).

## Running BlendingToolKit
- BlendingToolKit (btk) requires an input catalog that contains information required to simulate galaxies and blends.
This repository includes sample input catalogs with a small number of galaxies that can be used to draw blend images with btk. See [tutorials](https://github.com/LSSTDESC/BlendingToolKit/tree/master/notebooks) to learn how to run btk with these catalogs.
- CatSim Catalog corresponding to one square degree of sky and processed WeakLensingDeblending catalogs can be downloaded from [here](https://stanford.app.box.com/s/s1nzjlinejpqandudjyykjejyxtgylbk).
- [Cosmo DC2](https://arxiv.org/abs/1907.06530) catalog requires pre-processing in order to be used as input catalog to btk. Refer to this [notebook](https://github.com/LSSTDESC/WeakLensingDeblending/blob/cosmoDC2_ingestion/notebooks/wld_ingestion_cosmoDC2.ipynb) on how to convert the DC2 catalog into a CatSim-like catalog that can be analyzed with btk.

## Installation
For required packages, see [requirements.txt](https://github.com/LSSTDESC/BlendingToolKit/blob/master/requirements.txt)

More detailed installation instructions can be found [here](https://blendingtoolkit.readthedocs.io/en/latest/install.html).

## Contributing

See [CONTRIBUTING.md](https://github.com/LSSTDESC/BlendingToolKit/blob/master/CONTRIBUTING.md)
