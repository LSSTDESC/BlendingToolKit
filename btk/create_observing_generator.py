from btk.obs_conditions import DefaultObsConditions, all_surveys


class ObservingGenerator:
    def __init__(
        self,
        survey_name,
        obs_conds=None,
        verbose=False,
        stamp_size=24,
    ):
        """Generates class with observing conditions in each band.

        Args:
             survey_name (str): Name of the survey which should be available in descwl
             obs_conds: Class (not object) that returns observing conditions for
                             a given survey and band. If not provided, then the default
                             `descwl.survey.Survey` values for the corresponding
                             survey_name are used to create the observing_generator.
        """
        if type(survey_name) == str:
            self.surveys = [survey_name]
        elif type(survey_name) == list:
            self.surveys = survey_name
        else:
            raise TypeError("survey_name is not in the right format")
        self.verbose = verbose

        # create default observing conditions
        if obs_conds is None:
            self.obs_conds = DefaultObsConditions(stamp_size)
        else:
            assert obs_conds.stamp_size == stamp_size
            self.obs_conds = obs_conds

    def __iter__(self):
        return self

    def __next__(self):
        observing_generator = {}
        for s in self.surveys:
            observing_generator[s] = []
            for band in all_surveys[s]["bands"]:
                cutout = self.obs_conds(s, band)
                observing_generator[s].append(cutout)
        return observing_generator
