# BlendingToolKit
Tools to create blend catalogs, produce training samples and implement blending metrics

## Workflow
- Step 1: import raw catalogs from CatSim or the DC2 catalogs
- Step 2: modify those catalogs to create catalogs of (multiple) blends and explore joint distributions (separation, mangitude difference)
- Step 3: generate PSF-convolved images and perform data augmentation with tensorflow
- Step 4: perform blending-related tasks such as detection, segmentation, deblending and various measurements
- Step 5: test algorithms with a set of low- and high-level metrics

## Useful tools
- [WeakLensingDeblending](https://github.com/LSSTDESC/WeakLensingDeblending)
- [GalSim](https://github.com/GalSim-developers/GalSim/)
- [Scarlet](https://github.com/fred3m/scarlet/)
