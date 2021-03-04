import os
from abc import ABC
from abc import abstractmethod
from copy import deepcopy

import astropy.table
import numpy as np
import galsim


class Catalog(ABC):
    """Base class containing the catalog for BTK.
    Each different catalog should have a corresponding subclass of Catalog.

    Attributes:
        self.table (astropy.table) : Standardized table containing information from the catalog
        self._raw_catalog : Contains the raw catalog given by the user
    """

    def __init__(self, catalog, verbose=False):
        """Returns astropy table with catalog name from input class.

        Args:
            catalog : CatSim-like catalog or galsim COSMOS catalog to draw galaxies from.
            verbose: Whether to print information related to loading catalog.
        """
        self._raw_catalog = catalog
        self.verbose = verbose
        self.table = self._prepare_table()

        if self.verbose:
            print("Catalog loaded")

    @classmethod
    @abstractmethod
    def from_file(cls, catalog_file, verbose):
        """Constructs the catalog object from a file. Should be implemented in subclasses."""

    @abstractmethod
    def _prepare_table(self):
        """Carries operations to generate a standardized table.
        Should be implemented in subclasses."""

    @property
    def name(self):
        """Property containing the name of the (sub)class. Is used to check whether
        the catalog is compatible with the chosen DrawBlendsGenerator"""
        return self.__class__.__name__

    def get_raw_catalog(self):
        """Returns the raw catalog."""
        return self._raw_catalog


class CatsimCatalog(Catalog):
    """Implementation of Catalog for the Catsim catalog."""

    @classmethod
    def from_file(cls, catalog_file, verbose=False):
        """Constructs the catalog object from a file.
        Args:
            catalog_file: path to a file containing a readable astropy table
        """
        _, ext = os.path.splitext(catalog_file)
        fmt = "fits" if ext.lower() == ".fits" else "ascii.basic"
        catalog = astropy.table.Table.read(catalog_file, format=fmt)
        return cls(catalog, verbose=verbose)

    def _prepare_table(self):
        """Carries operations to generate a standardized table. Uses the preexisting
        astropy table and calculates some parameters of interest."""
        table = deepcopy(self._raw_catalog)

        # convert ra dec from degrees to arcsec in catalog.
        if "ra" in table.colnames:
            table["ra"] *= 3600
        if "dec" in table.colnames:
            table["dec"] *= 3600

        f = self._raw_catalog["fluxnorm_bulge"] / (
            self._raw_catalog["fluxnorm_disk"] + self._raw_catalog["fluxnorm_bulge"]
        )
        r_sec = np.hypot(
            self._raw_catalog["a_d"] * (1 - f) ** 0.5 * 4.66,
            self._raw_catalog["a_b"] * f ** 0.5 * 1.46,
        )
        # BTK now requires ref_mags, but Catsim still wants magnitudes
        table["ref_mag"] = self._raw_catalog["i_ab"]
        table["btk_size"] = r_sec
        # Adds the extra columns to both catalogs just to be sure - TO CHECK
        self._raw_catalog["ref_mag"] = self._raw_catalog["i_ab"]
        self._raw_catalog["btk_size"] = r_sec

        return table


class CosmosCatalog(Catalog):

    def __init__(self, catalog, galsim_catalog, verbose=False):
        super().__init__(catalog, verbose=verbose)
        self.galsim_catalog = galsim_catalog
    
    @classmethod
    def from_file(cls, catalog_files, verbose=False):
        """
        Constructs the catalog object from a file.

        Args:
            catalog_files: list containing the two paths to the COSMOS data
        """
        catalog_coord = astropy.table.Table.read(catalog_files[0])
        catalog_fit = astropy.table.Table.read(catalog_files[1])
        catalog = astropy.table.hstack([catalog_coord, catalog_fit])

        galsim_catalog = galsim.COSMOSCatalog(catalog_files[0])
        return cls(catalog, galsim_catalog, verbose=verbose)

    def _prepare_table(self):
        """Carries operations to generate a standardized table."""

        table = deepcopy(self._raw_catalog)
        # make notation for 'ra' and 'dec' standard across code.
        table.rename_column("RA", "ra")
        table.rename_column("DEC", "dec")
        table.rename_column("MAG", "ref_mag")
        #index = np.where(t["IDENT_1"] == self._raw_catalog["IDENT_1"] for t in table)
        index = np.arange(0, len(table))
        
        # convert ra dec from degrees to arcsec in catalog.
        table["ra"] *= 3600
        table["dec"] *= 3600

        size = self._raw_catalog["flux_radius"] * self._raw_catalog["PIXEL_SCALE"]
        table["btk_size"] = size
        table["btk_index"] = index
        # ADds the extra columns to both catalogs just to be sure
        self._raw_catalog["ref_mag"] = self._raw_catalog["MAG"]
        self._raw_catalog["btk_size"] = size
        return table

    def get_galsim_catalog(self):
        return self.galsim_catalog
