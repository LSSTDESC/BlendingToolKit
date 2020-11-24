from abc import ABC, abstractmethod
import descwl
import astropy.wcs as WCS


all_surveys = {
    "LSST": {
        "name": "LSST",
        "bands": ("y", "z", "i", "r", "g", "u"),
        "pixel_scale": 0.2,
    },
    "DES": {"name": "DES", "bands": ("i", "r", "g", "z"), "pixel_scale": 0.263},
    "CFHT": {"name": "CFHT", "bands": ("i", "r"), "pixel_scale": 0.185},
    "HSC": {"name": "HSC", "bands": ("y", "z", "i", "r", "g"), "pixel_scale": 0.17},
}


def make_wcs(pixel_scale, shape, center_pix=None, center_sky=None, projection=None):
    """Creates WCS for an image.
    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image in pixels.
        center_pix (tuple): position of the reference pixel used as the center of the
                            affine transform for the wcs.
        center_sky (list): sky coordinates corresponding to center_pix, in arcseconds
        projection(str): projection type, default to TAN. A list of available
                            types can be found in astropy.wcs documentation
    Returns:
        wcs: WCS
    """
    if center_pix is None:
        center_pix = [(s + 1) / 2 for s in shape]
    if center_sky is None:
        center_sky = [0 for _ in range(2)]
    if projection is None:
        projection = "TAN"
    w = WCS.WCS(naxis=2)
    w.wcs.ctype = ["RA---" + projection, "DEC--" + projection]
    w.wcs.crpix = center_pix
    w.wcs.cdelt = [pixel_scale / 3600 for _ in range(2)]
    w.wcs.crval = [c / 3600 for c in center_sky]
    w.array_shape = shape
    return w


class Cutout(ABC):
    def __init__(
        self, stamp_size, pixel_scale, center_pix=None, center_sky=None, projection=None
    ):
        """Class containing the necessary information to draw a postage stamp (PSF,
        pixel_scale, WCS, etc.) for a given survey and band.
        """
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.pix_stamp_size = int(self.stamp_size / pixel_scale)
        self.wcs = self.get_wcs(center_pix, center_sky, projection)

    def get_wcs(self, center_pix=None, center_sky=None, projection=None):
        return make_wcs(
            pixel_scale=self.pixel_scale,
            center_pix=center_pix,
            center_sky=center_sky,
            projection=projection,
            shape=(self.pix_stamp_size, self.pix_stamp_size),
        )


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
        **survey_kwargs: any arguments given to a descwl survey
    """

    def __init__(
        self,
        stamp_size,
        center_pix=None,
        center_sky=None,
        projection=None,
        **survey_kwargs,
    ):
        descwl.survey.Survey.__init__(self, **survey_kwargs)
        Cutout.__init__(
            self,
            stamp_size,
            survey_kwargs["pixel_scale"],
            center_pix,
            center_sky,
            projection,
        )

    def get_psf_sky(self, psf_stamp_size):
        """Returns postage stamp image of the PSF and mean background sky
        level value saved in the input obs_conds class
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
    @staticmethod
    def psf_function(r):
        # usually r = 0.3 * pix
        return galsim.Moffat(2, r)

    def get_psf(self, psf_stamp_size):
        assert psf_stamp_size % 2 == 1
        psf_int = self.psf_function(self.psf_args).withFlux(1.0)

        # Draw PSF
        psf = psf_int.drawImage(
            nx=self.psf_size,
            ny=self.psf_size,
            method="real_space",
            use_true_center=True,
            scale=self.pix,
        ).array

        # Make sure PSF vanishes on the edges of a patch that
        # has the shape of the initial npsf
        psf = psf - psf[0, int(self.psf_size / 2)] * 2
        psf[psf < 0] = 0
        psf = psf / np.sum(psf)

        return psf


class ObsConditions(ABC):
    def __init__(self, stamp_size=24):
        """Class that returns a cutout object for a given survey_name and band.
        If the information provided by this class is combined with the blend_catalogs,
        blend postage stamps can be drawn.
        Args:
            stamp_size (float): In arcseconds.
        """
        self.stamp_size = stamp_size

    @abstractmethod
    def __call__(self, survey_name, band):
        """
        Args:
            survey_name: Name of the survey which should be available in descwl
            band: filter name to get observing conditions for.
        Returns:
            A btk.cutout.Cutout object.
        """
        pass


class WLDObsConditions(ObsConditions):
    def __init__(self, stamp_size=24):
        """Returns the default observing conditions from the WLD package
        for a given survey_name and band.
        """
        super().__init__(stamp_size)

    def get_cutout_params(self, survey_name, band, pixel_scale):
        # get default survey params
        pix_stamp_size = int(self.stamp_size / pixel_scale)
        cutout_params = descwl.survey.Survey.get_defaults(
            survey_name=survey_name, filter_band=band
        )
        cutout_params["image_width"] = pix_stamp_size
        cutout_params["image_height"] = pix_stamp_size

        # Information for WCS
        cutout_params["center_sky"] = None
        cutout_params["center_pix"] = None
        cutout_params["projection"] = "TAN"

        return cutout_params

    def __call__(self, survey, band):
        pixel_scale = survey["pixel_scale"]
        cutout_params = self.get_cutout_params(survey["name"], band, pixel_scale)
        cutout = WLDCutout(
            self.stamp_size,
            no_analysis=True,
            survey_name=survey["name"],
            filter_band=band,
            **cutout_params,
        )

        if cutout.pixel_scale != pixel_scale:
            raise ValueError(
                "observing condition pixel scale does not "
                "match input pixel scale: {0} == {1}".format(
                    cutout.pixel_scale, pixel_scale
                )
            )
        if cutout.filter_band != band:
            raise ValueError(
                "observing condition band does not "
                "match input band: {0} == {1}".format(cutout.filter_band, band)
            )

        return cutout


class CosmosObsConditions(ObsConditions):
    def __init__(self):
        """Returns the default obs"""
        pass

    def __call__(self, survey, band):
        pixel_scale = survey["pixel_scale"]
        cutout_params = self.get_cutout_params(survey["name"], band, pixel_scale)
        cutout = WLDCutout(
            self.stamp_size,
            no_analysis=True,
            survey_name=survey["name"],
            filter_band=band,
            **cutout_params,
        )

        if cutout.pixel_scale != pixel_scale:
            raise ValueError(
                "observing condition pixel scale does not "
                "match input pixel scale: {0} == {1}".format(
                    cutout.pixel_scale, pixel_scale
                )
            )
        if cutout.filter_band != band:
            raise ValueError(
                "observing condition band does not "
                "match input band: {0} == {1}".format(cutout.filter_band, band)
            )

        return cutout
