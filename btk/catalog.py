"""Contains abstract base class `Catalog` that standarizes catalog usage across BTK."""
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
        """Creates Catalog object and standarizes raw_catalog information.

        The standarization is done via the attribute self.table via the _prepare_table method.

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
            raw_catalog["a_b"] * f ** 0.5 * 1.46,
        )
        # BTK now requires ref_mags, but Catsim still wants magnitudes
        table["ref_mag"] = raw_catalog["i_ab"]
        table["btk_size"] = r_sec

        return table


def apply_cosmos_exclusion(
    catalog, catalog_file, use_sample, exclusion_level, min_hlr=0, max_hlr=0, min_flux=0, max_flux=0
):
    """funtion to apply cuts on the cosmos catalogs.
    This function is heavily inspired from the galsim repo
    For more details refer to (https://galsim-developers.github.io/GalSim/_build/html/real_gal.html)

    Args:
        catalog(astropy.table.Table): Merged astropy table with real and parametric catalogs
        catalog_file(list): list containing the two paths to the COSMOS data. Please see
            the tutorial page for more details
            (https://lsstdesc.org/BlendingToolKit/tutorials.html#using-cosmos-galaxies).
        use_sample(str): sample to be used-options: "23.5", "25.2" magnitude cut
        exclusion_level(str): Level of additional cuts to make on the galaxies based on the quality
            of postage stamp definition and/or parametric fit quality [beyond the
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
            [default: "marginal"]
        min_hlr(float): Exclude galaxies whose fitted half-light radius is smaller than this
            value (in arcsec).  [default: 0, meaning no limit]
        max_hlr(float): Exclude galaxies whose fitted half-light radius is larger than this
            value (in arcsec).  [default: 0, meaning no limit]
        min_flux(float): Exclude galaxies whose fitted flux is smaller than this value.
            [default: 0, meaning no limit
        max_flux(float): Exclude galaxies whose fitted flux is larger than this value.
            [default: 0, meaning no limit]
    """
    mask = np.ones(len(catalog), dtype=bool)

    assert use_sample in ("23.5", "25.2"), 'The options for use_sample are: "23.5" and "25.2"'

    assert exclusion_level in (
        "none",
        "bad_stamp",
        "bad_fits",
        "marginal",
    ), 'The options for \
        exclusion_level are: "none", "bad_stamp", "bad_fits", "marginal"'

    if exclusion_level in ("marginal", "bad_stamp"):
        # First, read in what we need to impose selection criteria, if the appropriate
        # exclusion_level was chosen.

        # This should work if the user passed in (or we defaulted to) the real galaxy
        # catalog name:

        try:
            selection_file_name = catalog_file[0].replace(".fits", "_selection.fits")
            selection_cat = astropy.table.Table.read(selection_file_name)
        except FileNotFoundError:
            raise FileNotFoundError(
                "File with GalSim selection criteria not found. "
                "Run the program `galsim_download_cosmos -s %s` to get the "
                "necessary selection file. \n\
                the _selection.fits file must be present in the same repo as the real catalog, "
                'otherwise the "bad_stamp" and "marginal" cuts will fail' % (use_sample)
            )

        # We proceed to select galaxies in a way that excludes suspect postage stamps (e.g.,
        # with deblending issues), suspect parametric model fits, or both of the above plus
        # marginal ones.  These two options for 'exclusion_level' involve placing cuts on
        # the S/N of the object detection in the original postage stamp, and on issues with
        # masking that can indicate deblending or detection failures.  These cuts were used
        # in GREAT3.  In the case of the masking cut, in some cases there are messed up ones
        # that have a 0 for self.selection_cat['peak_image_pixel_count'].  To make sure we
        # don't divide by zero (generating a RuntimeWarning), and still eliminate those, we
        # will first set that column to 1.e-5.  We choose a sample-dependent mask ratio cut,
        # since this depends on the peak object flux, which will differ for the two samples
        # (and we can't really cut on this for arbitrary user-defined samples).
        if use_sample == "23.5":
            cut_ratio = 0.2
            sn_limit = 20.0
        else:
            cut_ratio = 0.8
            sn_limit = 12.0
        div_val = selection_cat["peak_image_pixel_count"]
        div_val[div_val == 0.0] = 1.0e-5
        mask &= (selection_cat["sn_ellip_gauss"] >= sn_limit) & (
            (selection_cat["min_mask_dist_pixels"] > 11.0)
            | (selection_cat["average_mask_adjacent_pixel_count"] / div_val < cut_ratio)
        )

        # Finally, impose a cut that the total flux in the postage stamp should be positive,
        # which excludes a tiny number of galaxies (of order 10 in each sample) with some sky
        # subtraction or deblending errors.  Some of these are eliminated by other cuts when
        # using exclusion_level='marginal'.
        if catalog is not None:
            mask &= catalog["stamp_flux"] > 0

    if exclusion_level in ("bad_fits", "marginal"):
        # This 'exclusion_level' involves eliminating failed parametric fits (bad fit status
        # flags).  In this case we only get rid of those with failed bulge+disk AND failed
        # Sersic fits, so there is no viable parametric model for the galaxy.
        sersicfit_status = catalog["fit_status"][:, 4]
        bulgefit_status = catalog["fit_status"][:, 0]
        mask &= ((sersicfit_status > 0) & (sersicfit_status < 5)) | (
            (bulgefit_status > 0) & (bulgefit_status < 5)
        )

    if exclusion_level == "marginal":
        # We have already placed some cuts (above) in this case, but we'll do some more.  For
        # example, a failed bulge+disk fit often indicates difficulty in fit convergence due to
        # noisy surface brightness profiles, so we might want to toss out those that have a
        # failure in EITHER fit.
        mask &= ((sersicfit_status > 0) & (sersicfit_status < 5)) & (
            (bulgefit_status > 0) & (bulgefit_status < 5)
        )

        # Some fit parameters can indicate a likely sky subtraction error: very high sersic n
        # AND abnormally large half-light radius (>1 arcsec).
        if "hlr" not in catalog.dtype.names:  # pragma: no cover
            raise OSError(
                "You still have the old COSMOS catalog.  Run the program "
                "`galsim_download_cosmos -s %s` to upgrade." % (use_sample)
            )
        hlr = catalog["hlr"][:, 0]
        n = catalog["sersicfit"][:, 2]
        mask &= (n < 5) | (hlr < 1.0)

        # Major flux differences in the parametric model vs. the COSMOS catalog can indicate fit
        # issues, deblending problems, etc.
        mask &= np.abs(selection_cat["dmag"]) < 0.8

    if min_hlr > 0.0 or max_hlr > 0.0 or min_flux > 0.0 or max_flux > 0.0:
        if "hlr" not in catalog.dtype.names:  # pragma: no cover
            raise OSError(
                "You still have the old COSMOS catalog.  Run the program "
                "`galsim_download_cosmos -s %s` to upgrade." % (use_sample)
            )

        hlr = catalog["hlr"][:, 0]  # sersic half-light radius
        flux = catalog["flux"][:, 0]

        if min_hlr > 0.0:
            mask &= hlr > min_hlr
        if max_hlr > 0.0:
            mask &= hlr < max_hlr
        if min_flux > 0.0:
            mask &= flux > min_flux
        if max_flux > 0.0:
            mask &= flux < max_flux

    return catalog[mask]


class CosmosCatalog(Catalog):
    """Class containing catalog information for drawing COSMOS galaxies from galsim."""

    def __init__(self, raw_catalog, galsim_catalog, verbose=False):
        """Initializes the COSMOS Catalog class."""
        super().__init__(raw_catalog, verbose=verbose)
        self.galsim_catalog = galsim_catalog

    @classmethod
    def from_file(
        cls,
        catalog_files,
        exclusion_level="marginal",
        use_sample="25.2",
        min_hlr=0,
        max_hlr=0,
        min_flux=0,
        max_flux=0,
        verbose=False,
    ):
        """Constructs the catalog object from a file. It also places exclusion levels and cuts.

        For more details refer to (https://galsim-developers.github.io/GalSim/_build/html/real_gal.html)

        Args:
            catalog_files(list): list containing the two paths to the COSMOS data. Please see
                the tutorial page for more details
                (https://lsstdesc.org/BlendingToolKit/tutorials.html#using-cosmos-galaxies).
            exclusion_level(str): Level of additional cuts to make on the galaxies based on the quality
                of postage stamp definition and/or parametric fit quality [beyond the
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
                Note that the _selection.fits file must be present in the same repo as catalog_files[0]
                real catalog (catalog_files[0]), Otherwise the "bad_stamp" and "marginal" cuts will fail
                [default: "marginal"]
            min_hlr(float): Exclude galaxies whose fitted half-light radius is smaller than this
                value (in arcsec).  [default: 0, meaning no limit]
            max_hlr(float): Exclude galaxies whose fitted half-light radius is larger than this
                value (in arcsec).  [default: 0, meaning no limit]
            min_flux(float): Exclude galaxies whose fitted flux is smaller than this value.
                [default: 0, meaning no limit]
            max_flux(float): Exclude galaxies whose fitted flux is larger than this value.
                [default: 0, meaning no limit]
            verbose: whether to print verbose info.
        """

        assert use_sample in ("23.5", "25.2"), 'The options for use_sample are: "23.5" and "25.2"'

        assert exclusion_level in (
            "none",
            "bad_stamp",
            "bad_fits",
            "marginal",
        ), 'The options for \
            exclusion_level are: "none", "bad_stamp", "bad_fits", "marginal"'

        catalog_coord = astropy.table.Table.read(catalog_files[0])
        catalog_fit = astropy.table.Table.read(catalog_files[1])
        catalog = astropy.table.hstack([catalog_coord, catalog_fit])
        catalog = apply_cosmos_exclusion(
            catalog=catalog,
            catalog_file=catalog_files,
            exclusion_level=exclusion_level,
            use_sample=use_sample,
            min_hlr=min_hlr,
            max_hlr=max_hlr,
            min_flux=min_flux,
            max_flux=0,
        )

        galsim_catalog = galsim.COSMOSCatalog(catalog_files[0], exclusion_level=exclusion_level)

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
