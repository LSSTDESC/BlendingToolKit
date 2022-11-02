"""Contains abstract base class `Catalog` that standarizes catalog usage across BTK."""
import os
from abc import ABC, abstractmethod
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
        """Creates Catalog object and standarizes raw_catalog information.

        The standarization is done via the attribute self.table via the _prepare_table method.

        Args:
            raw_catalog: Raw catalog containing information to create table.
            verbose: Whether to print information related to loading catalog.

        """
        self.verbose = verbose
        self.table = self._prepare_table(raw_catalog)
        self._raw_catalog = raw_catalog

        if self.verbose:
            print("Catalog loaded")

    @classmethod
    @abstractmethod
    def from_file(cls, catalog_file, verbose):
        """Constructs the catalog object from a file. Should be implemented in subclasses."""

    @abstractmethod
    def _prepare_table(self, raw_catalog):
        """Carries operations to generate a standardized table."""

    @property
    def name(self):
        """Property containing the name of the (sub)class.

        It is used to check whether the catalog is compatible with the chosen DrawBlendsGenerator.
        """
        return self.__class__.__name__

    def get_raw_catalog(self):
        """Returns the raw catalog."""
        return self._raw_catalog


class CatsimCatalog(Catalog):
    """Implementation of Catalog for the Catsim catalog."""

    @classmethod
    def from_file(cls, catalog_files, verbose=False):
        """Constructs the catalog object from a file.

        Args:
            catalog_files: path to a file containing a readable astropy table
            verbose (bool): Whether to print info.
        """
        _, ext = os.path.splitext(catalog_files)
        fmt = "fits" if ext.lower() == ".fits" else "ascii.basic"
        catalog = astropy.table.Table.read(catalog_files, format=fmt)
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
            raw_catalog["a_b"] * f**0.5 * 1.46,
        )
        # BTK now requires ref_mag, but Catsim still wants magnitudes
        table["ref_mag"] = raw_catalog["i_ab"]
        table["btk_size"] = r_sec

        return table


class CosmosCatalog(Catalog):
    """Class containing catalog information for drawing COSMOS galaxies from galsim."""

    def __init__(self, raw_catalog, galsim_catalog, verbose=False):
        """Initializes the COSMOS Catalog class."""
        super().__init__(raw_catalog, verbose=verbose)
        self.galsim_catalog = galsim_catalog

    @classmethod
    def from_file(cls, catalog_files, exclusion_level="marginal", verbose=False):
        """Constructs the catalog object from a file. It also places exclusion level cuts.

        For more details: (https://galsim-developers.github.io/GalSim/_build/html/real_gal.html)

        Args:
            catalog_files(list): list containing the two paths to the COSMOS data. Please see
                the tutorial page for more details
                (https://lsstdesc.org/BlendingToolKit/tutorials.html#using-cosmos-galaxies).
            exclusion_level(str): Level of additional cuts to make on the galaxies based on the
                quality of postage stamp definition and/or parametric fit quality [beyond the
                minimal cuts imposed when making the catalog - see Mandelbaum et
                al. (2012, MNRAS, 420, 1518) for details].
                Options:
                - "none": No cuts.
                - "bad_stamp": Apply cuts to eliminate galaxies that have failures in
                    postage stamp definition.  These cuts may also eliminate a small
                    subset of the good postage stamps as well.
                - "bad_fits": Apply cuts to eliminate galaxies that have failures in the
                    parametric fits.  These cuts may also eliminate a small
                    subset of the good parametric fits as well.
                - "marginal": Apply the above cuts, plus ones that eliminate some more
                    marginal cases.
                Note that the _selection.fits file must be present in the same repo as the real
                images catalog, Otherwise the "bad_stamp" and "marginal" cuts will fail
                [default: "marginal"]
            verbose: whether to print verbose info.
        """
        galsim_catalog = galsim.COSMOSCatalog(catalog_files[0], exclusion_level=exclusion_level)

        catalog_coord = astropy.table.Table.read(catalog_files[0])
        catalog_fit = astropy.table.Table.read(catalog_files[1])

        catalog_coord = catalog_coord[galsim_catalog.orig_index]
        catalog_fit = catalog_fit[galsim_catalog.orig_index]

        catalog = astropy.table.hstack([catalog_coord, catalog_fit])

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
        """Returns the galsim.COSMOSCatalog object."""
        return self.galsim_catalog


available_catalogs = {"catsim": CatsimCatalog, "cosmos": CosmosCatalog}
