import descwl


def default_obs_conditions(Args, band):
    """Returns the default observing conditions from the WLD package
    for a given survey_name and band.

    Args:
        Args: Class containing parameters to generate blends
        band: filter name to get observing conditions for.
    Returns:
        `descwl.survey.Survey`: WLD survey class with observing conditions.
    """
    survey = descwl.survey.Survey.get_defaults(
        survey_name=Args.survey_name, filter_band=band
    )
    return survey


def generate(Args, obs_function=None):
    """Generates class with observing conditions in each band.

    Args:
        Args: Class containing input parameters.
        obs_function: Function that outputs dict of observing conditions. If
            not provided then the default `descwl.survey.Survey` values for the
            corresponding Args.survey_name are used to create the
            observing_generator.

    Yields:
        Generator with `descwl.survey.Survey` class for each band.
    """
    while True:
        observing_generator = []
        for band in Args.bands:
            if obs_function:
                survey = obs_function(Args, band)
            else:
                survey = default_obs_conditions(Args, band)
                if Args.verbose:
                    print("Default observing conditions selected")
            survey["image_width"] = Args.stamp_size / survey["pixel_scale"]
            survey["image_height"] = Args.stamp_size / survey["pixel_scale"]
            descwl_survey = descwl.survey.Survey(
                no_analysis=True,
                survey_name=Args.survey_name,
                filter_band=band,
                **survey
            )
            if descwl_survey.pixel_scale != Args.pixel_scale:
                raise ValueError(
                    "observing condition pixel scale does not \
                    match input pixel scale: {0} == {1}".format(
                        descwl_survey.pixel_scale, Args.pixel_scale
                    )
                )
            if descwl_survey.filter_band != band:
                raise ValueError(
                    "observing condition band does not \
                    match input band: {0} == {1}".format(
                        descwl_survey.filter_band, band
                    )
                )
            observing_generator.append(descwl_survey)
        yield observing_generator
