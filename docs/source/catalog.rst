Input Catalog
===============

*BlendingToolKit* (btk) requires an input catalog that contains information required to simulate galaxies and blends. Each galaxy is parameterized as a bulge + disk  + agn with parameters following the `LSST DM catalog schema <https://confluence.lsstcorp.org/display/SIM/Database+Schema>`_


CatSim
-------
The catalog simulation framework (CatSim) is a database of astrophysical sources with properties that are representative of what the LSST will observe at its ten-year coadded depth. Refer to the official CatSim `page <https://www.lsst.org/scientists/simulations/catsim>`_ for more details.
The btk package includes a sample input catalog that contains parameters of 100 galaxies. A more extensive catalog can be downloaded from `here <https://stanford.box.com/s/s1nzjlinejpqandudjyykjejyxtgylbk>`_.


Cosmo DC2
---------
`CosmoDC2 <https://arxiv.org/abs/1907.06530>`_ is a large synthetic galaxy catalog designed to support precision dark energy science with the Large Synoptic Survey Telescope (LSST). Refer to this `notebook <https://github.com/LSSTDESC/WeakLensingDeblending/blob/cosmoDC2_ingestion/notebooks/wld_ingestion_cosmoDC2.ipynb>`_ on how to inject the DC2 catalog into a CatSim-like catalog that can be analyzed with btk. The btk package includes a sample input catalog that contains parameters of 15 blend scenes with 61 galaxies.
