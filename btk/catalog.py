import os
from abc import ABC
from abc import abstractmethod
from copy import deepcopy

import astropy.table
import galsim
import numpy as np


class Catalog(ABC):
    """Abstract base class containing the catalog for BTK.

    Each new catalog should be a subclass of Catalog.

    Attributes:
        self.table (astropy.table) : Standardized table containing information from the catalog
    """

    def __init__(self, raw_catalog, verbose=False):
        """Creates Catalog object and standarizes raw_catalog information into
        attribute self.table via _prepare_table method.

        Args:
            raw_catalog: Raw catalog containing information to create table.
            verbose: Whether to print information related to loading catalog.

        """
        self.verbose = verbose
        self.table = self._prepare_table(raw_catalog)

        if self.verbose:
            print("Catalog loaded")

    @classmethod
    @abstractmethod
    def from_file(cls, catalog_file, verbose):
        """Constructs the catalog object from a file. Should be implemented in subclasses."""

    @abstractmethod
    def _prepare_table(self, raw_catalog):
        """Carries operations to generate a standardized table. Should be implemented
        in subclasses."""

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

    def _prepare_table(self, raw_catalog):
        """Carries operations to generate a standardized table.

        Uses the preexisting astropy table and calculates some parameters of interest.
        """
        table = deepcopy(raw_catalog)

        # convert ra dec from degrees to arcsec in catalog.
        if "ra" in table.colnames:
            table["ra"] *= 3600
        if "dec" in table.colnames:
            table["dec"] *= 3600

        f = raw_catalog["fluxnorm_bulge"] / (
            raw_catalog["fluxnorm_disk"] + raw_catalog["fluxnorm_bulge"]
        )
        r_sec = np.hypot(
            raw_catalog["a_d"] * (1 - f) ** 0.5 * 4.66,
            raw_catalog["a_b"] * f ** 0.5 * 1.46,
        )
        # BTK now requires ref_mags, but Catsim still wants magnitudes
        table["ref_mag"] = raw_catalog["i_ab"]
        table["btk_size"] = r_sec

        return table


class CosmosCatalog(Catalog):
    def __init__(self, raw_catalog, galsim_catalog, verbose=False):
        super().__init__(raw_catalog, verbose=verbose)
        self.galsim_catalog = galsim_catalog

    @classmethod
    def from_file(cls, catalog_files, verbose=False):
        """Constructs the catalog object from a file.

        Args:
            catalog_files(list): list containing the two paths to the COSMOS data.
            verbose: whether to print verbose info.
        """
        catalog_coord = astropy.table.Table.read(catalog_files[0])
        catalog_fit = astropy.table.Table.read(catalog_files[1])
        catalog = astropy.table.hstack([catalog_coord, catalog_fit])
        galsim_catalog = galsim.COSMOSCatalog(catalog_files[0], exclusion_level="none")
        return cls(catalog, galsim_catalog, verbose=verbose)

    def _prepare_table(self, raw_catalog):
        """Carries operations to generate a standardized table."""
        table = deepcopy(raw_catalog)
        # make notation for 'ra' and 'dec' standard across code.
        table.rename_column("RA", "ra")
        table.rename_column("DEC", "dec")
        table.rename_column("MAG", "ref_mag")
        index = np.arange(0, len(table))

        # convert ra dec from degrees to arcsec in catalog.
        table["ra"] *= 3600
        table["dec"] *= 3600

        size = raw_catalog["flux_radius"] * raw_catalog["PIXEL_SCALE"]
        table["btk_size"] = size
        table["btk_index"] = index
        return table

    def get_galsim_catalog(self):
        return self.galsim_catalog
