"""Creates a generator of characteristics of a given observation for a given
survey.
ToDo:
Add options for variable PSF and seeing conditions
"""
import descwl


def default_obs_conditions(Args, band):
    """Returns the default observing conditions from the WLD package
    for a given survey_name and band
    Args
        Args: Class containing parameters to generate blends
        band: filter name to get observing conditions for.
    Returns
        survey: WLD survey class with observing conditions.
    """
    survey = descwl.survey.Survey.get_defaults(
        survey_name=Args.survey_name,
        filter_band=band)
    return survey


def generate(Args, obs_function=None):
    """Generates observing condition for each band.
    Args:
        Args: Class containing input parameters.
        obs_function: Function that outputs dict of observing conditions.
    Returns:
        Generator with descwl.survey.Survey class for each band.
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
            survey['image_width'] = Args.stamp_size / survey['pixel_scale']
            survey['image_height'] = Args.stamp_size / survey['pixel_scale']
            descwl_survey = descwl.survey.Survey(no_analysis=True,
                                                 survey_name=Args.survey_name,
                                                 filter_band=band, **survey)
            observing_generator.append(descwl_survey)
        yield observing_generator
