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
        self.multiresolution = multiresolution
        if multiresolution:
            self.survey_name = survey_name
            self.verbose = verbose

        else:
            if survey_name not in all_surveys:
                raise KeyError("Survey not implemented.")

            # self.bands = all_surveys[survey_name]["bands"]
            self.survey_name = survey_name
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
        if self.multiresolution:
            observing_generator = {}
            for s in self.survey_name:
                observing_generator[s] = []
                for band in all_surveys[s]["bands"]:
                    cutout = self.obs_conds(s, band)
                    observing_generator[s].append(cutout)
        else:
            observing_generator = []
            for band in self.bands:
                cutout = self.obs_conds(self.survey_name, band)
                observing_generator.append(cutout)
        return observing_generator
