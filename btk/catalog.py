import os
from copy import deepcopy
from abc import ABC, abstractmethod
import galsim
import astropy.table


class Catalog(ABC):
    def __init__(self, catalog, verbose=False):
        """Returns astropy table with catalog name from input class.

        Args:
            catalog : CatSim-like catalog or galsim COSMOS catalog to draw galaxies from.
            verbose: Whether to print information related to loading catalog.

        Attributes:
            self.table (`astropy.table`): CatSim-like catalog with selection criteria applied
                and recorded in the `_selection_functions` list.
        """
        self._raw_catalog = catalog
        self.verbose = verbose
        self.table = self._prepare_table()
        self._selection_functions = []

        if self.verbose:
            print("Catalog loaded")

    @classmethod
    @abstractmethod
    def from_file(cls, catalog_file, verbose):
        """Catalog constructor from input file"""
        pass

    @abstractmethod
    def _prepare_table(self):
        """Operations to standardize the catalog table"""
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def get_raw_catalog(self):
        return self._raw_catalog

    def apply_selection_function(self, selection_function):
        """Apply a selection cut to the current table.

        Parameters
        ----------
        selection_function: callable
            logical selection on the catalog table columns/rows.

        """
        if not callable(selection_function):
            raise TypeError("selection_function must be callable")

        self.table = selection_function(self.table)
        self._selection_functions.append(selection_function)


class WLDCatalog(Catalog):
    @classmethod
    def from_file(cls, catalog_file, verbose=False):
        # catalog returned is an astropy table.
        _, ext = os.path.splitext(catalog_file)
        fmt = "fits" if ext.lower() == ".fits" else "ascii.basic"
        catalog = astropy.table.Table.read(catalog_file, format=fmt)

        return cls(catalog, verbose=verbose)

    def _prepare_table(self):
        table = deepcopy(self._raw_catalog)

        # convert ra dec from degrees to arcsec in catalog.
        table["ra"] *= 3600
        table["dec"] *= 3600

        return table


class CosmosCatalog(Catalog):
    @classmethod
    def from_file(cls, catalog_file, verbose=False):
        # This will return a COSMOSCatalog object.
        catalog = galsim.COSMOSCatalog(file_name=catalog_file)

        return cls(catalog, verbose=verbose)

    def _prepare_table(self):
        table = astropy.table.Table(self._raw_catalog.real_cat)

        # make notation for 'ra' and 'dec' standard across code.
        table.rename_column("RA", "ra")
        table.rename_column("DEC", "dec")

        # convert ra dec from degrees to arcsec in catalog.
        table["ra"] *= 3600
        table["dec"] *= 3600

        return table
