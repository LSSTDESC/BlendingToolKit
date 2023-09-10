# BlendingToolKit

[![tests](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/pytest.yml)
[![notebooks](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/notebooks.yml/badge.svg?branch=main)](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/notebooks.yml)
[![docs](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/LSSTDESC/BlendingToolKit/actions/workflows/docs.yml)
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

## Workflow

<img src="docs/source/images/diagram.png" alt="btk workflow" width="550"/>

In red are components of the BTK pipeline that are intended to be easily customized by users to meet their
science needs.

## Code example

In what follows we illustrate how to use BTK to generate blended images, run a deblender on them, and
evaluate the performance of the deblender using metrics. For more details on this example see our
quick-start notebook at `notebooks/00-quickstart.ipynb`.

```python
import btk

# setup CATSIM catalog
catalog_name = "../data/input_catalog.fits"
catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)

# setup survey parameters
survey = btk.survey.get_surveys("LSST")

# setup sampling function
# this function determines how to organize galaxies in catalog into blends
stamp_size = 24.0
sampling_function = btk.sampling_functions.DefaultSampling(
    catalog=catalog, max_number=5, max_mag=25.3, stamp_size=stamp_size
)

# setup generator to create batches of blends
batch_size = 100
draw_generator = btk.draw_blends.CatsimGenerator(
    catalog, sampling_function, survey, batch_size, stamp_size
)

# get batch of blends
blend_batch = next(draw_generator)

# setup deblender (we use SEP in this case)
deblender = SepSingleBand(max_n_sources=5,
                          use_band=2 # measure on 'r' band
                          )

# run deblender on generated blend images (all batches)
deblend_batch = deblender(blend_batch)

# setup matcher
matcher = PixelHungarianMatcher(pixel_max_sep=5.0 # maximum separation in pixels for matching
)

# match true and predicted catalogs
truth_catalogs = blend_batch.catalog_list
pred_catalogs = deblend_batch.catalog_list
matching = matcher(truth_catalogs, pred_catalogs) # object with matching information

# compute detection performance on this batch
recall = btk.metrics.detection.Recall(batch_size)
precision = btk.metrics.detection.Precision(batch_size)
print("Recall: ", recall(matching.tp, matching.t, matching.p))
print("Precision: ", precision(matching.tp, matching.t, matching.p))
```

## Installation

BTK is pip installable, with the following command:

```bash
pip install blending-toolkit==1.0.0b2
```

In case of any issues and for details of required packages, please see the more detailed installation instructions [here](https://lsstdesc.org/BlendingToolKit/install.html).

## Contributing

Everyone can contribute to this project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) document for details.

In short, to interact with the project you can:

- Ask or Answer questions on the [Discussions Q&A page](https://github.com/LSSTDESC/BlendingToolKit/discussions).
- Report a bug by opening a [GitHub issue](https://github.com/LSSTDESC/BlendingToolKit/issues).
- Open a [GitHub issue](https://github.com/LSSTDESC/BlendingToolKit/issue) or [Discussions](https://github.com/LSSTDESC/BlendingToolKit/discussions) to ask for feedback on a planned contribution.
- Submit a [Pull Request](https://github.com/LSSTDESC/BlendingToolKit/pulls) to contribute to the code.

Issues marked with `contributions welcome` or `good first issue` are particularly good places to start. These are great ways to learn more about the inner workings of BTK.
