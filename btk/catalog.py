import os
from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod
import astropy.table


class Catalog(ABC):
    def __init__(self, catalog, verbose=False):
        """Returns astropy table with catalog name from input class.

        Args:
            catalog : CatSim-like catalog or galsim COSMOS catalog to draw galaxies from.
            verbose: Whether to print information related to loading catalog.

        Attributes:
            self.table (`astropy.table`): CatSim-like catalog.
        """
        self._raw_catalog = catalog
        self.verbose = verbose
        self.table = self._prepare_table()

        if self.verbose:
            print("Catalog loaded")

    @classmethod
    @abstractmethod
    def from_file(cls, catalog_file, verbose):
        """Catalog constructor from input file"""

    @abstractmethod
    def _prepare_table(self):
        """Operations to standardize the catalog table"""

    @property
    def name(self):
        return self.__class__.__name__

    def get_raw_catalog(self):
        return self._raw_catalog


class CatsimCatalog(Catalog):
    @classmethod
    def from_file(cls, catalog_file, verbose=False):
        # catalog returned is an astropy table.
        _, ext = os.path.splitext(catalog_file)
        fmt = "fits" if ext.lower() == ".fits" else "ascii.basic"
        catalog = astropy.table.Table.read(catalog_file, format=fmt)

        return cls(catalog, verbose=verbose)

    def _prepare_table(self):
        table = deepcopy(self._raw_catalog)

        # TODO: does the WLDCatalog require the 'ra' and 'dec' columns
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
        # BTK now requires ref_mags, but WLD still wants magnitudes
        table["ref_mag"] = self._raw_catalog["i_ab"]
        table["btk_size"] = r_sec
        # Adds the extra columns to both catalogs just to be sure
        self._raw_catalog["ref_mag"] = self._raw_catalog["i_ab"]
        self._raw_catalog["btk_size"] = r_sec

        return table


class CosmosCatalog(Catalog):
    @classmethod
    def from_file(cls, catalog_files, verbose=False):
        """
        Paramters
        ---------
        catalog_files: list of galsim cataolgs
        """
        catalog_coord = astropy.table.Table.read(catalog_files[0], "fits")
        catalog_fit = astropy.table.Table.read(catalog_files[1], "fits")
        catalog = astropy.table.hstack([catalog_coord, catalog_fit])

        return cls(catalog, verbose=verbose)

    def _prepare_table(self):
        table = deepcopy(self._raw_catalog)
        # make notation for 'ra' and 'dec' standard across code.
        table.rename_column("RA", "ra")
        table.rename_column("DEC", "dec")
        table.rename_column("MAG", "ref_mag")
        index = np.where(t["IDENT_1"] == self._raw_catalog["IDENT_1"] for t in table)

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
