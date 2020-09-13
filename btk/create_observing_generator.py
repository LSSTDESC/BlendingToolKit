from btk.obs_conditions import DefaultObsConditions, all_surveys


class ObservingGenerator:
    def __init__(
        self,
        survey_name,
        obs_conditions=None,
        verbose=False,
    ):
        """Generates class with observing conditions in each band.

        Args:
             survey_name (str): Name of the survey which should be available in descwl
             obs_conditions: Class (not object) that returns observing conditions for
                             a given survey and band. If not provided, then the default
                             `descwl.survey.Survey` values for the corresponding
                             survey_name are used to create the observing_generator.
        """
        if survey_name not in all_surveys:
            raise KeyError("Survey not implemented.")

        self.bands = all_surveys[survey_name]["bands"]
        self.survey_name = survey_name
        self.verbose = verbose

        # TODO: it might be a bit cumbersome for the user to create this dict.
        # create default observing conditions
        if obs_conditions is None:
            self.obs_conditions = DefaultObsConditions()
        else:
            self.obs_conditions = obs_conditions

    def __iter__(self):
        return self

    def __next__(self):
        observing_generator = []
        for band in self.bands:
            btk_survey = self.obs_conditions(self.survey_name, band)
            observing_generator.append(btk_survey)
        return observing_generator
