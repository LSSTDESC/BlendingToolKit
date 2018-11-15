"""Default settings for the Blending ToolKit"""


class Simulation_params(object):
    """Parametrs to create blends.
    Args:
        catalog_name: Name of input catalog from which to draw objects.
        max_number: Maximum number of objects per blend.
        batch_size: Number of blends to si,utae per batch.
        stamp_size: Size of postage stamp in arcseconds.
        survey_name: Name of survey to select observing conditions.
        seed: random seed.
        add_noise: If True, adds noise to output blended images.
        draw_isolated: If trues, drwas image sof each isolated object.
        bands: Filters in which to simulate images.
    """
    def __init__(self, catalog_name, max_number=2,
                 batch_size=8, stamp_size=24, psf_stamp_size=41,
                 survey_name="LSST",
                 seed=0, add_noise=True, draw_isolated=True,
                 bands=('u', 'g', 'r', 'i', 'z', 'y'),
                 verbose=False, draw_PSF=True):
        self.catalog_name = catalog_name
        self.max_number = max_number
        self.batch_size = batch_size
        self.stamp_size = stamp_size
        self.psf_stamp_size = psf_stamp_size
        self.survey_name = survey_name
        self.add_noise = add_noise
        self.draw_isolated = draw_isolated
        self.seed = seed
        self.bands = bands
        self.verbose = verbose
        self.draw_PSF = draw_PSF
