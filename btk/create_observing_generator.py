import btk.survey
import descwl
import astropy.wcs as WCS


def make_wcs(
    pixel_scale, shape, center_pix=None, center_sky=None, projection=None, naxis=2
):
    """Creates wcs for an image
    
    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image
        center_pix (tuple): position of the reference pixel used as the center of the
                            affine transform for the wcs.
        center_sky (float):
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


def default_obs_conditions(survey_name, band):
    """Returns the default observing conditions from the WLD package
    for a given survey_name and band.

    Args:
        survey_name: Name of the survey which should be available in descwl
        band: filter name to get observing conditions for.
    Returns:
        `survey`: Dictionary containing the observing conditions and WCS information.
    """
    survey = descwl.survey.Survey.get_defaults(
        survey_name=survey_name, filter_band=band
    )
    survey["center_sky"] = None
    survey["center_pix"] = None
    survey["projection"] = "TAN"
    return survey


class ObservingGenerator:
    def __init__(
        self,
        survey_name,
        stamp_size,
        obs_function=default_obs_conditions,
        verbose=False,
    ):
        """Generates class with observing conditions in each band.

        Args:
             obs_function: Function that outputs dict of observing conditions. If
                           not provided, then the default `descwl.survey.Survey` values
                           for the corresponding survey_name are used to create the
                           observing_generator.
        """
        if survey_name not in btk.survey.surveys:
            raise KeyError("Survey not implemented.")
        self.survey_name = survey_name
        self.bands = btk.survey.surveys["survey_name"]["bands"]
        self.pixel_scale = btk.survey.surveys["survey_name"]["pixel_scale"]
        self.stamp_size = stamp_size
        self.obs_function = obs_function
        self.verbose = verbose

    def __next__(self):
        observing_generator = []
        for band in self.bands:
            # TODO: Add verbose if we used default observing conditions.
            survey = self.obs_function(self.survey_name, band)
            survey["image_width"] = self.stamp_size / survey["pixel_scale"]
            survey["image_height"] = self.stamp_size / survey["pixel_scale"]
            stamp_size = int(self.stamp_size / self.pixel_scale)
            wcs = make_wcs(
                pixel_scale=self.pixel_scale,
                center_pix=survey["center_pix"],
                center_sky=survey["center_sky"],
                projection=survey["projection"],
                shape=(stamp_size, stamp_size),
            )
            btk_survey = btk.survey.Survey(
                no_analysis=True,
                survey_name=self.survey_name,
                filter_band=band,
                wcs=wcs,
                **survey
            )
            if btk_survey.pixel_scale != self.pixel_scale:
                raise ValueError(
                    "observing condition pixel scale does not "
                    "match input pixel scale: {0} == {1}".format(
                        btk_survey.pixel_scale, self.pixel_scale
                    )
                )
            if btk_survey.filter_band != band:
                raise ValueError(
                    "observing condition band does not "
                    "match input band: {0} == {1}".format(btk_survey.filter_band, band)
                )
            observing_generator.append(btk_survey)
        return observing_generator

    def __iter__(self):
        return self
