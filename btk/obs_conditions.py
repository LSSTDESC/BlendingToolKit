from abc import ABC, abstractmethod
import astropy.wcs as WCS
import numpy as np
import galsim
import descwl
from collections import namedtuple

# A simple class archetype to serve as a dictionary without having to write the field names
# every time.
Survey = namedtuple(
    "Survey",
    [
        "name",
        "pixel_scale",
        "filters",
    ],
)

Filter = namedtuple(
    "Filter",
    [
        "name",
        "psf_scale",
        "mean_sky_level",
        "exp_time",
        "zero_point",
    ],
)


pix_ROMAN = 0.11
pix_RUBIN = 0.2
pix_HST = 0.06
pix_EUCLID = 0.101
pix_HSC = 0.167
pix_DES = 0.263
pix_CFHT = 0.185

# Sigma of the psf profile in arcseconds.
# https://arxiv.org/pdf/1702.01747.pdf Z-band
sigma_ROMAN = 0.11 * np.array([1.68, 1.69, 1.86, 2.12, 2.44, 2.71])

# https://www.lsst.org/about/camera/features
sigma_RUBIN = np.array([0.327, 0.31, 0.297, 0.285, 0.276, 0.267])

# https://sci.esa.int/documents/33859/36320/1567253682555-Euclid_presentation_Paris_1Dec2009.pdf
sigma_EUCLID = np.array([0.16])

# Source https://hst-docs.stsci.edu/display/WFC3IHB/6.6+UVIS+Optical+Performance#id-6
# .6UVISOpticalPerformance-6.6.1 800nm
sigma_HST = np.array([0.074])

# https://hsc-release.mtk.nao.ac.jp/doc/ deep+udeep
sigma_HSC = np.array([0.306, 0.285, 0.238, 0.268, 0.272])


# Euclid = Survey(
#     "Euclid",
#     pix_EUCLID,
#     sigma_EUCLID,
#     ["VIS"],
#     np.array([22.9]),
#     np.array([2260]),
#     np.array([6.85]),
# )
# HST = Survey(
#     "HST",
#     pix_HST,
#     sigma_HST,
#     ["f814w"],
#     np.array([22]),
#     np.array([3000]),
#     np.array([20]),
# )
# HSC = Survey(
#     "HSC",
#     pix_HSC,
#     sigma_HSC,
#     ["g", "r", "i", "z", "y"],
#     np.array([21.4, 20.6, 19.7, 18.3, 17.9]),
#     np.array([600, 600, 1200, 1200, 1200]),
#     np.array([91.11, 87.74, 69.80, 29.56, 21.53]),
# )
# Roman = Survey(
#     "Roman",
#     pix_ROMAN,
#     sigma_ROMAN,
#     ["F062", "Z087", "Y106", "J129", "H158", "F184"],
#     np.array([22, 22, 22, 22, 22, 22]),  ## Not Checked!!!
#     np.array([3000, 3000, 3000, 3000, 3000, 3000]),  ## Not Checked!!!
#     np.array([26.99, 26.39, 26.41, 26.35, 26.41, 25.96]),
# )
Rubin = Survey(
    "LSST",
    pix_RUBIN,
    [Filter("y",0.327,18.6,4800,10.58),
     Filter("z",0.310,19.6,4800,22.68),
     Filter("i",0.297,20.5,5520,32.36),
     Filter("r",0.285,21.2,5520,43.70),
     Filter("g",0.276,22.3,2400,50.70),
     Filter("u",0.267,22.9,1680,9.16),
    ]
)
# DES = Survey("DES", pix_DES, None, ["i", "r", "g", "z"], None, None, None)
# CFHT = Survey("CFHT", pix_CFHT, None, ["i", "r"], None, None, None)


def make_wcs(pixel_scale, shape, center_pix=None, center_sky=None, projection="TAN"):
    """Creates WCS for an image.
    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image in pixels.
        center_pix: tuple representing the center of the image in pixels
        center_sky: tuple representing the center of the image in sky coordinates
                     (RA,DEC) in arcseconds.
        projection(str): projection type, default to TAN. A list of available
                            types can be found in astropy.wcs documentation
    Returns:
        wcs: WCS
    """
    if center_pix is None:
        center_pix = [(s + 1) / 2 for s in shape]
    if center_sky is None:
        center_sky = [0 for _ in range(2)]
    w = WCS.WCS(naxis=2)
    w.wcs.ctype = ["RA---" + projection, "DEC--" + projection]
    w.wcs.crpix = center_pix
    w.wcs.cdelt = [pixel_scale / 3600 for _ in range(2)]
    w.wcs.crval = [c / 3600 for c in center_sky]
    w.array_shape = shape
    return w


class Cutout(ABC):
    def __init__(self, stamp_size, pixel_scale, **wcs_kwargs):
        """Class containing the necessary information to draw a postage stamp (PSF,
        pixel_scale, WCS, etc.) for a given survey and band.

        Parameters
        ----------
        stamp_size: float
            stamp size in arcsec
        pixel_scale: float
            pixel scale in arcsec
        wcs_kwargs: arguments for generating wcs.
        """
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.pix_stamp_size = int(self.stamp_size / pixel_scale)
        shape = (self.pix_stamp_size, self.pix_stamp_size)
        self.wcs = make_wcs(pixel_scale, shape, **wcs_kwargs)


class WLDCutout(descwl.survey.Survey, Cutout):
    """Extension of the descwl survey class including information for the WCS.
    Args:
        center_pix: tuple representing the center of the image in pixels
        center_sky: tuple representing the center of the image in sky coordinates
                     (RA,DEC) in arcseconds.
        projection: string representing the type of projection for the WCS. If None,
                     it will default to "TAN". A list of available projections can
                     be found in the documentation of `astropy.wcs`
        wcs: an `astropy.wcs.wcs` object corresponding to the parameters center_pix,
              center_sky, projection, pixel_scale and stamp_size.
        survey_kwargs: any arguments given to a descwl survey
    """

    def __init__(
        self,
        stamp_size,
        pixel_scale,
        survey_kwargs,
        **wcs_kwargs,
    ):
        descwl.survey.Survey.__init__(self, **survey_kwargs)
        Cutout.__init__(self, stamp_size, pixel_scale, **wcs_kwargs)

    def get_psf_sky(self, psf_stamp_size):
        """Returns postage stamp image of the PSF and mean background sky
        level value.

        Args:
            psf_stamp_size: Size of postage stamp to draw PSF on in pixels.
        Returns:
            psf_image (np.ndarray): Postage stamp image of PSF
            mean_sky_level (float): Mean of sky background
        """
        mean_sky_level = self.mean_sky_level
        psf = self.psf_model
        psf_image = psf.drawImage(
            scale=self.pixel_scale, nx=psf_stamp_size, ny=psf_stamp_size
        ).array
        return psf_image, mean_sky_level


class CosmosCutout(Cutout):
    def __init__(
        self,
        stamp_size,
        pixel_scale,
        filt,
        psf_stamp_size=41,
    ):
        """Class containing the necessary information to draw a postage stamp (PSF,
        pixel_scale, WCS, etc.) using Cosmos' real galaxies for a given survey.

        Parameters
        ----------
        stamp_size: int
            size of the image cutout in pixels
        survey: `Survey`
            a `Survey` object that contains the meta-information for the cutout
        psf_stamp_size: int
            size of the psf stamp in pixels

        """
        super(CosmosCutout, self).__init__(stamp_size, pixel_scale)
        self.psf_stamp_size = psf_stamp_size
        self.filt = filt
        self.band = filt.name

    @staticmethod
    def psf_function(r):
        return galsim.Moffat(2, r)

    def get_psf(self):
        """Generates a psf as a Galsim object using the survey information"""
        assert self.psf_stamp_size % 2 == 1

        psf_obj = self.psf_function(self.filt.psf_scale).withFlux(1.0)
        psf = psf_obj.drawImage(
            nx=self.psf_stamp_size,
            ny=self.psf_stamp_size,
            method="no_pixel",
            use_true_center=True,
            scale=self.pixel_scale,
        ).array

        # Make sure PSF vanishes on the edges of a patch that
        # has the shape of the initial psf
        psf = psf - psf[1, int(self.psf_stamp_size / 2)] * 2
        psf[psf < 0] = 0
        psf /= np.sum(psf)
        # Generating an unintegrated galsim psf for the convolution
        psf_obj = galsim.InterpolatedImage(
            galsim.Image(psf), scale=self.pixel_scale
        ).withFlux(1.0)

        return psf_obj


class ObsConditions(ABC):
    def __init__(self, stamp_size=24, psf_stamp_size=8.2):
        """Class that returns a cutout object for a given survey_name and band.
        If the information provided by this class is combined with the blend_catalogs,
        blend postage stamps can be drawn.
        Args:
            stamp_size (float): In arcseconds.
            psf_stamp_size (float): In arcseconds.
        """
        self.stamp_size = stamp_size
        self.psf_stamp_size = psf_stamp_size

    @abstractmethod
    def __call__(self, survey_name, filt):
        """
        Args:
            survey_name: Name of the survey which should be available in descwl
            band: filter name to get observing conditions for.
        Returns:
            A btk.cutout.Cutout object.
        """
        pass


class WLDObsConditions(ObsConditions):
    """Returns the default observing conditions from the WLD package
    for a given survey_name and band.
    """

    def __call__(self, survey, filt):
        pix_stamp_size = int(self.stamp_size / survey.pixel_scale)

        # get parameters for the descwl.Survey.survey object.
        survey_kwargs = descwl.survey.Survey.get_defaults(
            survey_name=survey.name, filter_band=filt.name
        )
        survey_kwargs["zero_point"] = filt.zero_point
        survey_kwargs["exposure_time"] = filt.exp_time
        # Ideally we would make use of filt.psf_scale and filt.mean_sky_level but those are not directly inputs of WLD, so it needs some thinking.
        survey_kwargs["image_width"] = pix_stamp_size
        survey_kwargs["image_height"] = pix_stamp_size
        survey_kwargs["no_analysis"] = True
        survey_kwargs["survey_name"] = survey.name
        survey_kwargs["filter_band"] = filt.name

        cutout = WLDCutout(self.stamp_size, survey.pixel_scale, survey_kwargs)

        if cutout.pixel_scale != survey.pixel_scale:
            raise ValueError(
                "observing condition pixel scale does not "
                f"match input pixel scale: {cutout.pixel_scale} == {survey.pixel_scale}"
            )
        if cutout.filter_band != filt.name:
            raise ValueError(
                "observing condition band does not "
                f"match input band: {cutout.filter_band} == {band}"
            )

        return cutout


class CosmosObsConditions(ObsConditions):
    def __call__(self, survey, filt):
        psf_stamp_size = int(self.psf_stamp_size / survey.pixel_scale)
        while psf_stamp_size % 2 == 0:
            psf_stamp_size += 1

        return CosmosCutout(self.stamp_size, survey.pixel_scale, filt, psf_stamp_size)
