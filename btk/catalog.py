import os
from copy import deepcopy
from abc import ABC, abstractmethod
import galsim
import astropy.table


class Catalog(ABC):
    def __init__(self, catalog_file, selection_function=lambda x: x, verbose=False):
        """Returns astropy table with catalog name from input class.

        Args:
            catalog_file: File path of CatSim-like catalog or galsim COSMOS catalog to
                         draw galaxies from.
            selection_function: Selection cuts (if input) to place on input catalog.
            verbose: Whether to print information related to loading catalog.

        Attributes:
            self.table (`astropy.table`) : CatSim-like catalog with a selection
                                           criteria applied if provided.
        """
        self.cat = self.get_catalog(catalog_file)
        self.table = selection_function(self.get_table())
        self.verbose = verbose

        if self.verbose:
            print("Catalog loaded")

    @abstractmethod
    def get_catalog(self, catalog_file):
        pass

    @abstractmethod
    def get_table(self):
        pass

    @abstractmethod
    @property
    def name(self):
        # class name
        pass


class WLDCatalog(Catalog):
    def get_catalog(self, catalog_file):
        _, ext = os.path.splitext(catalog_file)
        fmt = "fits" if ext == ".fits" else "ascii.basic"
        cat = astropy.table.Table.read(catalog_file, format=fmt)
        return cat

    def get_table(self):
        table = deepcopy(self.cat)

        # convert ra dec from degrees to arcsec in catalog.
        table["ra"] *= 3600
        table["dec"] *= 3600

        return self.cat

    @property
    def name(self):
        return "WLDCatalog"


class CosmosCatalog(Catalog):
    def get_catalog(self, catalog_file):
        return galsim.COSMOSCatalog(file_name=catalog_file)

    def get_table(self):
        table = astropy.table.Table(self.cat.real_cat)

        # make notation for 'ra' and 'dec' standard across code.
        table.rename_column("RA", "ra")
        table.rename_column("DEC", "dec")

        # convert ra dec from degrees to arcsec in catalog.
        table["ra"] *= 3600
        table["dec"] *= 3600
        return table

    @property
    def name(self):
        return "CosmosCatalog"
