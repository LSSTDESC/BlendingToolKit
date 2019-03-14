class Simulation_params(object):
    """Parameters to create blends.

    Attributes:
        catalog_name: Name of input catalog from which to draw objects.
        max_number: Maximum number of objects per blend.
        batch_size: Number of blends to simulate per batch.
        stamp_size: Size of postage stamp in arcseconds.
        survey_name: Name of survey to select observing conditions.
        seed: random seed.
        add_noise: If True, adds noise to output blended images.
        bands: Filters in which to simulate images.
        min_snr(float): Simulate signals from individual sources down to this
            S/N threshold, where the signal N is calculated for the full
            exposure time and the noise N is set by the expected fluctuations
            in the sky background during a full exposure.
        verbose: If true prints returns description at multiple steps.
    """
    def __init__(self, catalog_name, max_number=2,
                 batch_size=8, stamp_size=24,
                 survey_name="LSST",
                 seed=0, add_noise=True, draw_isolated=True,
                 bands=('u', 'g', 'r', 'i', 'z', 'y'), min_snr=0.05,
                 verbose=False):
        """Inits Simulation_params with input observing conditions and image
        parametrs."""
        self.catalog_name = catalog_name
        self.max_number = max_number
        self.batch_size = batch_size
        self.stamp_size = stamp_size
        self.survey_name = survey_name
        self.add_noise = add_noise
        self.seed = seed
        self.bands = bands
        self.min_snr = min_snr
        self.verbose = verbose
        if survey_name is "LSST":
            self.pixel_scale = 0.2
        elif survey_name is "DES":
            self.pixel_scale = 0.263
        elif survey_name is "CFHT":
            self.pixel_scale = 0.185
        elif survey_name is "HSC":
            self.pixel_scale = 0.17
        else:
            raise Exception(
                "survey_name should be LSST, DES, CFHT or HSC. Input name was \
                 {0}".format(survey_name))
