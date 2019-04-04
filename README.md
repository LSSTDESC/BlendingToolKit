[![Build Status](https://travis-ci.org/LSSTDESC/BlendingToolKit.svg?branch=master)](https://travis-ci.org/LSSTDESC/BlendingToolKit)
[![Documentation Status](https://readthedocs.org/projects/blendingtoolkit/badge/?version=latest)](https://blendingtoolkit.readthedocs.io/en/latest/?badge=latest)

# BlendingToolKit
Tools to create blend catalogs, produce training samples and implement blending metrics

Documentation can be found at https://blendingtoolkit.readthedocs.io/en/latest/

## Workflow
- Step 1: import raw catalogs from CatSim or the DC2 catalogs(not yet implemented)
- Step 2: modify those catalogs to create catalogs of (multiple) blends and explore joint distributions (separation, magnitude difference)
- Step 3: generate PSF-convolved images and perform data augmentation
- Step 4: train deep-learning algorithms to do various blending-related tasks such as detection, segmentation, deblending and measurements
- Step 5: test algorithms with a set of low- and high-level performance metrics


## Running BlendingToolKit
- BlendingToolKit (btk) requires an input catalog that contains information required to simulate galaxies and blends.
This repository includes sample input catalogs with small number of galaxies that can be used to draw blend images with btk. See [tutorials](https://github.com/LSSTDESC/BlendingToolKit/tree/master/notebooks) to learn how to run btk with these catalogs.
- Catalog corresponding to one square degree sky and pre-processed WeakLensingDeblending catalogs can be downloaded from [here](https://stanford.app.box.com/s/s1nzjlinejpqandudjyykjejyxtgylbk).

## Requirements
To run btk you need to install
- [WeakLensingDeblending](https://github.com/LSSTDESC/WeakLensingDeblending)
- [GalSim](https://github.com/GalSim-developers/GalSim/)
- numpy
- astropy
- fitsio
- scipy
- lmfit

More detailed installation instructions can be found [here](https://blendingtoolkit.readthedocs.io/en/latest/install.html)

### Optional
The tutorials include examples of using btk with some packages like
- [Scarlet](https://github.com/fred3m/scarlet/) (multi-band deblender)
- [sep](https://sep.readthedocs.io/en/v1.0.x/index.html)(Source Extraction and Photometry)
- [lsst](https://pipelines.lsst.io) science pipeline
