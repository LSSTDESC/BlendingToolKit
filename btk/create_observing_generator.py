"""Creates a generator of characteristics of a given observation for a given
survey.
"""
import descwl


def generate(Args):
    while True:
        if Args.survey_name is 'LSST':
            observing_generator = []
            for band in Args.bands:
                survey = descwl.survey.Survey.get_defaults(survey_name='LSST',
                                                           filter_band=band)
                descwl_survey = descwl.survey.Survey(survey_name='LSST',
                                                     filter_band='u', **survey)
                observing_generator.append(descwl_survey)
        yield observing_generator
