import btk.survey
from btk.obs_conditions import DefaultObsConditions


class ObservingGenerator:
    def __init__(
        self,
        survey_name,
        stamp_size,
        obs_conditions=DefaultObsConditions,
        verbose=False,
    ):
        """Generates class with observing conditions in each band.

        Args:
             survey_name (str): Name of the survey which should be available in descwl
             obs_conditions: Class (not object) that returns observing conditions for
                             a given survey and band. If not provided, then the default
                             `descwl.survey.Survey` values for the corresponding
                             survey_name are used to create the observing_generator.
             stamp_size: In arcseconds.
        """
        if survey_name not in btk.survey.surveys:
            raise KeyError("Survey not implemented.")

        self.survey_name = survey_name
        self.bands = btk.survey.surveys["survey_name"]["bands"]
        self.stamp_size = stamp_size

        self.obs_conditions = obs_conditions
        self.verbose = verbose

    def __iter__(self):
        return self

    def __next__(self):
        observing_generator = []
        for band in self.bands:
            obs_function = self.obs_conditions(self.survey_name, band)
            btk_survey = obs_function()
            observing_generator.append(btk_survey)
        return observing_generator
