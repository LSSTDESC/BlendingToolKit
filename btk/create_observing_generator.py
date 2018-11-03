"""Creates a generator of characteristics of a given observation for a given
survey.
ToDo:
Add options for variabe psf and seeing conditions
"""
import descwl


def default_obs_conditions(Args, band):
    survey = descwl.survey.Survey.get_defaults(
        survey_name=Args.survey_name,
        filter_band=band)
    return survey


def generate(Args, obs_function=None):
    while True:
        observing_generator = []
        for band in Args.bands:
            if obs_function:
                survey = obs_function(Args, band)
            else:
                survey = default_obs_conditions(Args, band)
            survey['image_width'] = Args.stamp_size / survey['pixel_scale']
            survey['image_height'] = Args.stamp_size / survey['pixel_scale']
            descwl_survey = descwl.survey.Survey(survey_name=Args.survey_name,
                                                 filter_band='u', **survey)
            observing_generator.append(descwl_survey)
        yield observing_generator
