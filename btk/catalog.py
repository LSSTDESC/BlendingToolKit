"""Contains abstract base class `Catalog` that standarizes catalog usage across BTK."""
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, Union

import astropy
import galsim
import numpy as np
from astropy.table import Table


class Catalog(ABC):
    """Abstract base class containing the catalog for BTK.

    Each new catalog should be a subclass of Catalog.

    Attributes:
        self.table (astropy.table) : Standardized table containing information from the catalog
    """

    def __init__(self, raw_catalog: Table):
        """Creates Catalog object and standarizes raw_catalog information.

        The standarization is done via the attribute self.table via the _prepare_table method.

        Args:
            raw_catalog: Raw catalog containing information to create table.

        """
        self.table = self._prepare_table(raw_catalog)
        self._raw_catalog = raw_catalog

    @classmethod
    @abstractmethod
    def from_file(cls, catalog_files: Union[str, Tuple[str, str]]):
        """Constructs the catalog object from a file. Should be implemented in subclasses."""

    @abstractmethod
    def _prepare_table(self, raw_catalog: Table):
        """Carries operations to generate a standardized table."""

    @property
    def name(self) -> str:
        """Property containing the name of the (sub)class.

        It is used to check whether the catalog is compatible with the chosen DrawBlendsGenerator.
        """
        return self.__class__.__name__

    def get_raw_catalog(self) -> Table:
        """Returns the raw catalog."""
        return self._raw_catalog


class CatsimCatalog(Catalog):
    """Implementation of Catalog for the Catsim catalog."""

    @classmethod
    def from_file(cls, catalog_files: str):
        """Constructs the catalog object from a file.

        Args:
            catalog_files: path to a file containing a readable astropy table
        """
        _, ext = os.path.splitext(catalog_files)
        fmt = "fits" if ext.lower() == ".fits" else "ascii.basic"
        catalog = Table.read(catalog_files, format=fmt)
        return cls(catalog)

    def _prepare_table(self, raw_catalog: Table):
        """Carries operations to generate a standardized table.

        Uses the preexisting astropy table and calculates some parameters of interest.
        """
        table = deepcopy(raw_catalog)
        if "ra" not in table.colnames or "dec" not in table.colnames:
            raise ValueError("Catalog must have 'ra' and 'dec' columns.")
        return table


class CosmosCatalog(Catalog):
    """Class containing catalog information for drawing COSMOS galaxies from galsim."""

    def __init__(self, raw_catalog: Table, galsim_catalog: galsim.COSMOSCatalog):
        """Initializes the COSMOS Catalog class."""
        super().__init__(raw_catalog)
        self.galsim_catalog = galsim_catalog

    @classmethod
    def from_file(cls, catalog_files: Tuple[str, str], exclusion_level="marginal"):
        """Constructs the catalog object from a file. It also places exclusion level cuts.

        For more details: (https://galsim-developers.github.io/GalSim/_build/html/real_gal.html)

        Args:
            catalog_files: tuple containing the two paths to the COSMOS data.
            exclusion_level: Level of additional cuts to make on the galaxies based on the
                quality of postage stamp definition and/or parametric fit quality (beyond the
                minimal cuts imposed when making the catalog - see ``Mandelbaum et
                al. (2012, MNRAS, 420, 1518)`` for details).
                Options:

                    - "none": No cuts.
                    - "bad_stamp": Apply cuts to eliminate galaxies that have failures in
                        postage stamp definition. These cuts may also eliminate a small
                        subset of the good postage stamps as well.
                    - "bad_fits": Apply cuts to eliminate galaxies that have failures in the
                        parametric fits. These cuts may also eliminate a small subset of the good
                        parametric fits as well.
                    - "marginal": Apply the above cuts, plus ones that eliminate some more
                        marginal cases.

                Note that the _selection.fits file must be present in the same repo as the real
                images catalog, Otherwise the "bad_stamp" and "marginal" cuts will fail
                (default: "marginal")
        """
        galsim_catalog = galsim.COSMOSCatalog(catalog_files[0], exclusion_level=exclusion_level)

        catalog_coord = Table.read(catalog_files[0])
        catalog_fit = Table.read(catalog_files[1])

        catalog_coord = catalog_coord[galsim_catalog.orig_index]
        catalog_fit = catalog_fit[galsim_catalog.orig_index]

        catalog = astropy.table.hstack([catalog_coord, catalog_fit])

        return cls(catalog, galsim_catalog)

    def _prepare_table(self, raw_catalog: Table) -> Table:
        """Carries operations to generate a standardized table."""
        table = deepcopy(raw_catalog)
        # make notation for 'ra' and 'dec' standard across code.
        table.rename_column("RA", "ra")
        table.rename_column("DEC", "dec")
        index = np.arange(0, len(table))
        table["btk_index"] = index

        return table

    def get_galsim_catalog(self) -> galsim.COSMOSCatalog:
        """Returns the galsim.COSMOSCatalog object."""
        return self.galsim_catalog


available_catalogs = {"catsim": CatsimCatalog, "cosmos": CosmosCatalog}
