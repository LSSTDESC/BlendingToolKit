# BlendingToolKit

![tests](https://github.com/LSSTDESC/BlendingToolKit/workflows/tests/badge.svg)
![tests](https://github.com/LSSTDESC/BlendingToolKit/workflows/docs/badge.svg)
[![notebooks](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/notebooks.yml/badge.svg?branch=main)](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/notebooks.yml)
[![codecov](https://codecov.io/gh/LSSTDESC/BlendingToolKit/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/BlendingToolKit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPI][pypi-badge]][pypi]

[pypi-badge]: <https://img.shields.io/pypi/pyversions/blending-toolkit?color=yellow&logo=pypi>
[pypi]: <https://pypi.org/project/blending-toolkit/>

## Summary

Framework for fast generation and analysis of galaxy blends catalogs. This toolkit is a convenient way of
producing multi-band postage stamp images of blend scenes and evaluate the performance of deblending algorithms.

Documentation can be found at <https://lsstdesc.org/BlendingToolKit/index.html>.

To get started with BTK check at our quickstart notebook at: `notebooks/00-quickstart.ipynb`

## Workflow

<img src="docs/source/images/diagram.png" alt="btk workflow" width="550"/>

- In red are the BTK objects that can be customized in various ways by BTK users.

## Installation

BTK is pip installable, with the following command:

```bash
pip install blending_toolkit
```

In case of any issues and for details of required packages, please see the more detailed installation instructions [here](https://lsstdesc.org/BlendingToolKit/install.html).

## Contributing

See our contributing guidelines at [CONTRIBUTING.md](https://github.com/LSSTDESC/BlendingToolKit/blob/main/CONTRIBUTING.md)
