"""Creates a generator of characteristics of a given observation for a given
survey.
"""
import descwl


def generate(Args):
    while True:
        observing_generator = []
        for band in Args.bands:
            survey = descwl.survey.Survey.get_defaults(
                survey_name=Args.survey_name,
                filter_band=band)
            survey['image_width'] = Args.stamp_size / survey['pixel_scale']
            survey['image_height'] = Args.stamp_size / survey['pixel_scale']
            descwl_survey = descwl.survey.Survey(survey_name=Args.survey_name,
                                                 filter_band='u', **survey)
            observing_generator.append(descwl_survey)
    yield observing_generator
