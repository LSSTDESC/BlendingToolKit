from btk.obs_conditions import WLDObsConditions, Survey


class ObservingGenerator:
    def __init__(
        self,
        surveys,
        obs_conds=None,
        verbose=False,
        stamp_size=24,
    ):
        """Generates class with observing conditions in each band.

        Args:
             surveys (`Survey` or `list`): btk.obs_condition.Survey object or a list of surveys surveys. See
                                    obs_conditions.py for a list of available surveys.
             obs_conds: Class (not object) that returns observing conditions for
                             a given survey and band. If not provided, then the default
                             `descwl.survey.Survey` values for the corresponding
                             survey_name are used to create the observing_generator.
        """
        if isinstance(surveys, Survey):
            self.surveys = [surveys]
        elif isinstance(surveys, list):
            self.surveys = []
            for s in surveys:
                if isinstance(s, Survey):
                    self.surveys.append(s)
                else:
                    raise TypeError(
                        "surveys should be a `btk.obs_conditions.Survey` object"
                    )
        else:
            raise TypeError("surveys should be a `btk.obs_conditions.Survey` object")

        self.verbose = verbose

        # create default observing conditions
        if obs_conds is None:
            self.obs_conds = WLDObsConditions(stamp_size)
        else:
            if not obs_conds.stamp_size == stamp_size:
                raise ValueError(
                    "Observing conditions stamp_size does not match "
                    "stamp_size given."
                )
            self.obs_conds = obs_conds

    def __iter__(self):
        return self

    def __next__(self):
        observing_generator = {}
        for s in self.surveys:
            observing_generator[s.name] = []
            for filt in s.filters:
                cutout = self.obs_conds(s, filt)
                observing_generator[s.name].append(cutout)
        return observing_generator
