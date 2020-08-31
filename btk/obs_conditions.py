from abc import ABC, abstractmethod

import astropy.wcs as WCS
import descwl

import btk.survey


def make_wcs(
    pixel_scale, shape, center_pix=None, center_sky=None, projection=None, naxis=2
):
    """Creates wcs for an image

    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image
        center_pix (tuple): position of the reference pixel used as the center of the
                            affine transform for the wcs.
        center_sky (list):
        naxis (int):
        projection(str):

    Returns:
        wcs: WCS
    """
    if center_pix is None:
        center_pix = [(s + 1) / 2 for s in shape]
    if center_sky is None:
        center_sky = [0 for _ in range(naxis)]
    if projection is None:
        projection = "TAN"
    w = WCS.WCS(naxis=2)
    w.wcs.ctype = ["RA---" + projection, "DEC--" + projection]
    w.wcs.crpix = center_pix
    w.wcs.cdelt = [pixel_scale for _ in range(naxis)]
    w.wcs.crval = center_sky
    w.array_shape = shape
    return w


class WLDObsConditions(ABC):
    def __init__(self, survey_name, band, stamp_size):
        """Class that returns observing conditions for a given survey_name and band.
        If the information provided by this class is combined with the blend_catalogs,
        blend postage stamps can be drawn.

        Args:
            survey_name: Name of the survey which should be available in descwl
            band: filter name to get observing conditions for.
        """
        self.survey_name = survey_name
        self.band = band
        self.stamp_size = stamp_size
        self.pixel_scale = btk.survey.surveys[survey_name]["pixel_scale"]
        self.pix_stamp_size = int(self.stamp_size / self.pixel_scale)

    @abstractmethod
    def get_survey(self):
        """Returns a btk.survey.Survey object."""
        pass

    def __call__(self):

        btk_survey = self.get_survey()

        if btk_survey.pixel_scale != self.pixel_scale:
            raise ValueError(
                "observing condition pixel scale does not "
                "match input pixel scale: {0} == {1}".format(
                    btk_survey.pixel_scale, self.pixel_scale
                )
            )
        if btk_survey.filter_band != self.band:
            raise ValueError(
                "observing condition band does not "
                "match input band: {0} == {1}".format(btk_survey.filter_band, self.band)
            )

        return btk_survey


class DefaultObsConditions(WLDObsConditions):
    def __init__(self, survey_name, band, stamp_size=24):
        """Returns the default observing conditions from the WLD package
        for a given survey_name and band.
        """
        super().__init__(survey_name, band, stamp_size)

    def get_survey(self):
        # get default survey params
        survey_params = descwl.survey.Survey.get_defaults(
            survey_name=self.survey_name, filter_band=self.band
        )
        survey_params["image_width"] = self.pix_stamp_size
        survey_params["image_height"] = self.pix_stamp_size

        # Information for WCS
        survey_params["center_sky"] = None
        survey_params["center_pix"] = None
        survey_params["projection"] = "TAN"

        wcs = make_wcs(
            pixel_scale=self.pixel_scale,
            center_pix=survey_params["center_sky"],
            center_sky=survey_params["center_pix"],
            projection=survey_params["projection"],
            shape=(self.pix_stamp_size, self.pix_stamp_size),
        )
        btk_survey = btk.survey.Survey(
            no_analysis=True,
            survey_name=self.survey_name,
            filter_band=self.band,
            wcs=wcs,
            **survey_params
        )

        return btk_survey
