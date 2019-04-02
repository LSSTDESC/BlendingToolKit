[![Build Status](https://travis-ci.org/LSSTDESC/BlendingToolKit.svg?branch=master)](https://travis-ci.org/LSSTDESC/BlendingToolKit)
# BlendingToolKit
Tools to create blend catalogs, produce training samples and implement blending metrics

Documentation can be found at https://blendingtoolkit.readthedocs.io/en/latest/

## Workflow
- Step 1: import raw catalogs from CatSim or the DC2 catalogs
- Step 2: modify those catalogs to create catalogs of (multiple) blends and explore joint distributions (separation, magnitude difference)
- Step 3: generate PSF-convolved images and perform data augmentation
- Step 4: train deep-learning algorthms to do various blending-related tasks such as detection, segmentation, deblending and measurements
- Step 5: test algorithms with a set of low- and high-level performance metrics


## Input Catalog
- Catsim like catalogs and pre-processed WeakLensingDeblending catlogs can be found [here](https://weaklensingdeblending.readthedocs.io/en/latest/products.html)

## Useful tools
- [WeakLensingDeblending](https://github.com/LSSTDESC/WeakLensingDeblending)
- [GalSim](https://github.com/GalSim-developers/GalSim/)
- [Scarlet](https://github.com/fred3m/scarlet/)
