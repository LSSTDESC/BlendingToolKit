"""For an input measure generator and measurement_function algorithm
this script will:
1) compare the output from measure to truth
2) return performance metrics

Performance metrics computed separately for detection, segmentation, flux,
redshift

"""
import numpy as np
import btk.config
import btk.get_input_catalog
import btk.create_blend_generator
import btk.create_observing_generator
import btk.measure
import btk.draw_blends
from scipy import spatial


class Metrics_params(object):
    def make_meas_generator(self, catalog_name, max_number=2,
                            batch_size=1, seed=999):
        """
        Creates the default btk.meas_generator for input catalog
        Overwrite this function for user defined measurement generator
        """
        # Load parameters
        param = btk.config.Simulation_params(
            catalog_name, max_number=max_number, batch_size=batch_size,
            seed=seed)
        np.random.seed(param.seed)
        # Load input catalog
        catalog = btk.get_input_catalog.load_catlog(param)
        # Generate catalogs of blended objects
        blend_generator = btk.create_blend_generator.generate(
            param, catalog)
        # Generates observing conditions
        observing_generator = btk.create_observing_generator.generate(
            param)
        # Generate images of blends in all the observing bands
        draw_blend_generator = btk.draw_blends.generate(
            param, blend_generator, observing_generator)
        meas_params = btk.measure.Measurement_params()
        meas_generator = btk.measure.generate(
            meas_params, draw_blend_generator, param)
        self.meas_generator = meas_generator

    def get_detections(self, index):

        return [], []

    def get_segmentation(self, data=None, index=None):
        return None

    def get_flux(self, data=None, index=None):
        return None

    def get_redshift(self, data=None, index=None):
        return None


def evaluate_detection(detected_centers, true_centers,
                       distance_upper_bound=4):
    """
    Compares the true centers and detected centers to identify the
    number of true detections, number of sources that were undetected
    and number of spurious detections.
    Args:
        true_catalog: Catalog with true parameters of galaxies in blend.
        detected_centers: Centroids from detection algorithm ([N, (x, y)])
        distance_upper_bound: Match objects within this distance (in pixels)
                              (Default:10)
    Returns:
    """
    if len(detected_centers) == 0:
        # no detection
        return 0, len(true_centers), 0
    z_tree = spatial.KDTree(true_centers)
    detected_centers = np.array(detected_centers).reshape(-1, 2)
    match = z_tree.query(detected_centers,
                         distance_upper_bound=distance_upper_bound)
    fin, = np.where(match[0] != np.inf)  # match exists
    inf, = np.where(match[0] == np.inf)  # no match within distance_upper_bound
    detected = len(np.unique(match[1][fin]))
    undetected = len(np.setdiff1d(range(len(true_centers)), match[1][fin]))
    spurious = len(np.unique(match[1][inf]))
    return detected, undetected, spurious


def evaluate_segmentation(segmentation, data=None, index=None):
    if segmentation is None:
        return None
    return None


def evaluate_flux(flux, data=None, index=None):
    if flux is None:
        return None
    return None


def evaluate_redshift(redshift, data=None, index=None):
    if redshift is None:
        return None
    return None


def run(Metrics_params, test_size=1000):
    results = {'detection': [], 'segmentation': [],
               'flux': [], 'redshift': []}
    for i in range(test_size):
        # Evaluate detection algorithm
        detected_centers, true_centers = Metrics_params.get_detections(
            index=i)
        results['detection'].append(evaluate_detection(
            detected_centers, true_centers))
        # Evaluate segmentation algorithm
        segmentation = Metrics_params.get_segmentation(index=i)
        results['segmentation'].append(evaluate_segmentation(
            segmentation))
        # Evaluate flux measurement algorithm
        flux = Metrics_params.get_flux(
            index=i)
        results['flux'].append(evaluate_flux(
            flux))
        # Evaluate redshift estimation algorithm
        redshift = Metrics_params.get_redshift(
            index=i)
        results['redshift'].append(evaluate_redshift(
            redshift))
    return results
