"""For an input measure generator and measurement_function algorithm
this script will:
1) compare the output from measure to truth
2) return performance metrics

Performance metrics computed separately for detection, segmentation, flux,
redshift

"""
import numpy as np
from scipy import spatial


class Metrics_params(object):
    def get_detections(self, data=None, index=None):
        return None

    def get_segmentation(self, data=None, index=None):
        return None

    def get_flux(self, data=None, index=None):
        return None

    def get_redshift(self, data=None, index=None):
        return None


def evaluate_detection(detected_centers, data, index,
                       distance_upper_bound=3):
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
    true_catalog = data[0]['blend_list'][index]
    if detected_centers is None:
        return [None, None, None]
    peaks = np.stack([true_catalog['dx'], true_catalog['dy']]).T
    z_tree = spatial.KDTree(peaks)
    match = z_tree.query(detected_centers,
                         distance_upper_bound=distance_upper_bound)
    detected = len(np.unique(match[1]))
    undetected = len(np.setdiff1d(range(len(true_catalog)), match))
    detected = len(np.unique(match[1]))
    spurious = len(np.unique(match[1][match[0] == np.inf]))
    return [detected, undetected, spurious]


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


def generate(Metrics_params, measure_generator, Args):
    while True:
        meas_output = next(measure_generator)
        # meas_output = blend_output, deblend_results, measured_results
        batch_size = len(meas_output[0]['blend_images'])
        results = {}
        for i in range(batch_size):
            results[i] = {}
            # Evaluate detection algorithm
            detected_centers = Metrics_params.get_detection(
                data=meas_output, index=i)
            results[i]['detection'] = evaluate_detection(
                detected_centers, data=meas_output, index=i)
            # Evaluate segmentation algorithm
            segmentation = Metrics_params.get_segmentation(
                data=meas_output, index=i)
            results[i]['segmentation'] = evaluate_segmentation(
                segmentation, data=meas_output, index=i)
            # Evaluate flux measurement algorithm
            flux = Metrics_params.get_flux(
                data=meas_output, index=i)
            results[i]['flux'] = evaluate_flux(
                flux, data=meas_output, index=i)
            # Evaluate redshift estimation algorithm
            redshift = Metrics_params.get_redshift(
                data=meas_output, index=i)
            results[i]['redshift'] = evaluate_redshift(
                redshift, data=meas_output, index=i)
        yield results, meas_output
