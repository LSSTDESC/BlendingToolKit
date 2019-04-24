"""For an input measure generator and measurement_function algorithm
this script will:
1) compare the output from measure to truth
2) return performance metrics

Performance metrics computed separately for detection, segmentation, flux,
redshift

"""
import numpy as np
import astropy.table


class Metrics_params(object):
    def __init__(self, meas_generator, param):
        """Class describing functions to return results of
         detection/deblending/measurement algorithm in meas_generator. Each
         blend results yielded by the meas_generator for a batch.
    """
        self.meas_generator = meas_generator
        self.param = param

    def get_detections(self):
        """Overwrite this function to return results from the detection algorithm.

        Returns:
            Results of the detection algorithm are returned as:
                true_tables:  List of astropy Tables of the blend catalogs of the
                    batch. Length of tables must be the batch size. x and y coordinate
                    values must be under columns named 'dx' and 'dy' respectively, in
                    pixels from bottom left corner as (0, 0).
                detected_tables: List of astropy Tables of output from detection
                    algorithm. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'dx' and 'dy'
                    respectively, in pixels from bottom left corner as (0, 0).
        """
        # Astropy table with entries corresponding to true sources
        true_tables = [astropy.table.Table()] * self.param.batch_size
        # Astropy table with entries corresponding to detections
        detected_centers = [np.array((2,))] * self.param.batch_size
        return true_tables, detected_centers

    def get_segmentation(self):
        """Define function here to return results from the segmentation
        algorithm.
        """
        return None

    def get_flux(self):
        """Define function here to return results from the flux measurement
         algorithm.
        """
        return None

    def get_shapes(self):
        """Define function here to return results from the shape measurement
         algorithm.
        """
        return None


def get_detction_match(true_table, detected_table):
    """Match detections to true objects and update values in the input
    blend catalog and detection catalog.

    """

    t_x = true_table['dx'][:, np.newaxis] - detected_table['dx']
    t_y = true_table['dy'][:, np.newaxis] - detected_table['dy']
    dist = np.hypot(t_x, t_y)
    norm_size = true_table['size']
    norm_dist = dist/norm_size[:, np.newaxis]
    detected_table['closest_gal_id'] = np.argmin(dist, axis=0)
    detected_table['closest_dist'] = np.min(dist, axis=0)
    detected_table['norm_closest_dist'] = np.min(norm_dist, axis=0)
    condlist = [np.min(norm_dist, axis=0) <= 1, np.min(norm_dist, axis=0) > 1]
    choicelist = [np.argmin(norm_dist, axis=0), -1]
    detected_table['match_id'] = np.select(condlist, choicelist)
    detected_table['is_spurious'] = np.zeros(len(detected_table), dtype=int)
    detected_table['is_spurious'][detected_table['match_id'] == -1] = 1
    true_table['num_dectections'] = 0
    true_table['closest_det_id'] = np.argmin(dist, axis=1)
    for j in detected_table['match_id']:
        if j > -1:
            true_table['num_dectections'][j] += 1
    true_table['is_shred'] = np.zeros(len(true_table), dtype=int)
    true_table['is_shred'][true_table['num_dectections'] > 1] = 1


def evaluate_detection(true_tables, detected_tables, batch_index):
    """
    Compares the true centers and detected centers to identify the
    number of true detections, number of sources that were undetected
    and number of spurious detections.
    Args:
        true_tables:  List of astropy Tables of the blend catalogs of the
            batch. Length of tables must be the batch size. x and y coordinate
            values must be under columns named 'dx' and 'dy' respectively, in
            pixels from bottom left corner as (0, 0).
        detected_tables: List of astropy Tables of output from detection
            algorithm. Length of tables must be the batch size. x and y
            coordinate values must be under columns named 'dx' and 'dy'
            respectively, in pixels from bottom left corner as (0, 0).
        batch_index(int): Index number of the batch.
    Returns:
    """
    batch_true_table = astropy.table.Table()
    batch_detected_table = astropy.table.Table()
    batch_blend_list = []
    for i in range(len(true_tables)):
        true_table = true_tables[i]
        num_true = len(true_table)
        true_table['true_id'] = range(num_true)
        true_table['blend_index'] = batch_index + i
        true_table['batch_index'] = batch_index
        det_table = detected_tables[i]
        num_det = len(det_table)
        det_table['detection_id'] = range(num_det)
        det_table['blend_index'] = batch_index + i
        det_table['batch_index'] = batch_index
        get_detction_match(true_table, det_table)
        batch_detected_table = astropy.table.vstack(
            (batch_detected_table, det_table))
        batch_true_table = astropy.table.vstack(
            (batch_true_table, true_table))
        detected_true = det_table['match_id'][det_table['match_id'] >= 0]
        num_detected = len(np.unique(detected_true))
        num_undetected = len(np.where(true_table['num_dectections'] == 0)[0])
        num_spurious = len(np.where(det_table['is_spurious'] == 1)[0])
        num_shred = len(np.where(true_table['is_shred'] == 1)[0])
        assert num_detected + num_undetected == num_true, "Number of "\
            "detected objects + number undetected objects must be equal to "\
            "the total number of true objects"
        num_matched_detections = true_table['num_dectections'].sum()
        assert num_matched_detections + num_spurious == num_det, "Number of "\
            "detections match to a true object + number of spurious must be "\
            "equal to the total number of detections."
        batch_blend_list.append([num_true, num_detected, num_undetected,
                                 num_spurious, num_shred])
    return batch_true_table, batch_detected_table, batch_blend_list


def evaluate_segmentation(segmentation, data=None, index=None):
    if segmentation is None:
        return None
    return None


def evaluate_flux(flux, data=None, index=None):
    if flux is None:
        return None
    return None


def evaluate_shapes(shapes, data=None, index=None):
    if shapes is None:
        return None
    return None


def run(Metrics_params, test_size=1000):
    results = {'detection': [astropy.table.Table(),
                             astropy.table.Table(),
                             []],
               'segmentation': [],
               'flux': [], 'shapes': []}
    for i in range(test_size):
        # Evaluate detection algorithm
        batch_detection_result = Metrics_params.get_detections()
        true_table, detected_table, blend_detection_list = evaluate_detection(
            batch_detection_result[0], batch_detection_result[1],
            batch_index=i)
        results['detection'][0] = astropy.table.vstack(
            [results['detection'][0], true_table])
        results['detection'][1] = astropy.table.vstack(
            [results['detection'][1], detected_table])
        results['detection'][2].extend(blend_detection_list)
        # Evaluate segmentation algorithm
        segmentation = Metrics_params.get_segmentation()
        results['segmentation'].append(evaluate_segmentation(
            segmentation, index=i))
        # Evaluate flux measurement algorithm
        flux = Metrics_params.get_flux()
        results['flux'].append(evaluate_flux(
            flux, index=i))
        # Evaluate shape measurement algorithm
        shapes = Metrics_params.get_shapes()
        results['shapes'].append(evaluate_shapes(
            shapes, index=i))
    return results
