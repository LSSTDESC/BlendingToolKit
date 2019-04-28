"""For an input measure generator and measurement_function algorithm
this script will:
1) compare the output from measure to truth
2) return performance metrics

Performance metrics computed separately for detection, segmentation, flux,
redshift

"""
import numpy as np
import astropy.table
import scipy.spatial


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
        detected_tables = [astropy.table.Table()] * self.param.batch_size
        return true_tables, detected_tables

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


def get_closest_neighbor_distance(true_table):
    """Returns a astropy.table.column with the distance to the closestobject.

    Function uses scipy.apatial to compute distance between the object centers.
    If object is the only one in the blend then the cosest_dist value is set to
    np.inf.

    Args:
        true_table: Catalog with entries corresponding to one blend.

    Returns:
        `astropy.table.Column`s: size of the galaxy.
    """
    peaks = np.stack(
        [np.array(true_table['dx']), np.array(true_table['dy'])]).T
    if peaks.shape[0] > 1:
        distance = scipy.spatial.distance.cdist(peaks, peaks,
                                                metric='euclidean')
        min_dist = [
            np.min(distance[i][distance[i] > 0]) for i in range(len(distance))]
        true_table['min_dist'] = min_dist


def initialize_detection_tables(detected_table, true_table,
                                batch_index, batch_size,
                                blend_index):
    """Initialize column entries of true objects and detection catalog to their
    default values.

    This is necessary since, if either true objects or detection table is
    empty, then there is no match in get_detection_match and those column
    entries would not otherwise be present in the tables.

    Function does not return anything, only the astropy tables are updates.
    """
    # initialize true objects table columns
    num_true = len(true_table)
    true_table['true_id'] = range(num_true)  # id in blend [0-num_true]
    # index of blend in test size [0 - len(blend_summary)]
    true_table['blend_index'] = np.ones(
        num_true, dtype=int)*(batch_index*batch_size + blend_index)
    # index of batch in test size [0 - batch_size]
    true_table['batch_index'] = np.ones(
        num_true, dtype=int)*batch_index
    # number of times object was detected [0 - num_det]
    true_table['num_dectections'] = np.zeros(num_true, dtype=int)
    # id of closest detection; [0 - num_det] if detected, else -1
    true_table['closest_det_id'] = -1 * np.ones(num_true, dtype=int)
    # if multiple detections are made within size of object flag set to 1
    true_table['is_shred'] = np.zeros(num_true, dtype=int)
    # find distance to nearest neighbor. If isolated then set to np.inf
    true_table['min_dist'] = np.ones(num_true)*np.inf
    get_closest_neighbor_distance(true_table)
    # initialize detected objects table columns
    num_det = len(detected_table)
    detected_table['detection_id'] = range(num_det)  # id in blend [0-num_det]
    # index of blend in test size [0 - len(blend_summary)]
    detected_table['blend_index'] = np.ones(
        num_det, dtype=int)*(batch_index*batch_size + blend_index)
    # index of batch in test size [0 - batch_size]
    detected_table['batch_index'] = np.ones(
        num_det, dtype=int)*batch_index
    # id of closest true object; [0 - num_true] if detected, else -1
    detected_table['match_id'] = np.ones(
        num_det, dtype=int)*-1
    # if no match with true object flag set to 1
    detected_table['is_spurious'] = np.zeros(len(detected_table), dtype=int)


def get_detection_match(true_table, detected_table):
    """Match detections to true objects and update values in the input
    blend catalog and detection catalog.

    Function does not return anything, only the astropy tables are updates.

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        detected_table(astropy.table.Table): Table with entries corresponding
            to output of measurement algorithm in one blend.

    """
    if (len(detected_table) == 0 or len(true_table) == 0):
        # No match since either no detection or no true objects
        return
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
    detected_table['is_spurious'][detected_table['match_id'] == -1] = 1
    np.testing.assert_array_equal(
        np.argmin(dist, axis=1), np.argmin(norm_dist, axis=1),
        err_msg='norm_dist computation is wrong.')
    true_table['closest_det_id'] = np.argmin(dist, axis=1)
    for j in detected_table['match_id']:
        if j > -1:
            true_table['num_dectections'][j] += 1
    true_table['is_shred'][true_table['num_dectections'] > 1] = 1


def get_blend_detection_summary(true_table, det_table):
    """Returns list summarizing results of detection metric computation

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        detected_table(astropy.table.Table): Table with entries corresponding
            to output of measurement algorithm in one blend.

    Returns:
        List of detection metrics summary:
            num_true: number of true objects.
            num_detected: Number of correct detections by algorithm.
            num_undetected: Number of true objects not detected by algorithm.
            num_spurious: Number of spurious detections.
            num_shred: Number of true objects shredded by algorithm.

    """
    num_true = len(true_table)
    num_det = len(det_table)
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
    blend_summary = [num_true, num_detected, num_undetected,
                     num_spurious, num_shred]
    return blend_summary


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
    batch_size = len(true_tables)
    for i in range(len(true_tables)):
        true_table = true_tables[i]
        detected_table = detected_tables[i]
        # initialize columns to default values
        initialize_detection_tables(detected_table, true_table,
                                    batch_index, batch_size, i)
        # match detection and true source
        get_detection_match(true_table, detected_table)
        # summarize blend detection results to table
        blend_summary = get_blend_detection_summary(true_table, detected_table)
        batch_blend_list.append(blend_summary)
        # add results to batch table
        batch_detected_table = astropy.table.vstack(
            (batch_detected_table, detected_table))
        batch_true_table = astropy.table.vstack(
            (batch_true_table, true_table))
    # check table
    np.testing.assert_array_equal(batch_true_table['is_shred'] == 1,
                                  batch_true_table['num_dectections'] > 1,
                                  err_msg='Shredded object has less than 2'
                                  'detections')
    np.testing.assert_array_equal(batch_detected_table['is_spurious'] == 1,
                                  batch_detected_table['match_id'] == -1,
                                  err_msg='Spurious detection has match')
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
        if (
            len(batch_detection_result[0]) != len(batch_detection_result[1]) or
            len(batch_detection_result[0]) != Metrics_params.param.batch_size
           ):
            raise ValueError("Metrics_params.get_detections output must be "
                             "two lists of astropy table of length batch size."
                             f" Found {len(batch_detection_result[0])}, "
                             f"{len(batch_detection_result[1])}, "
                             f"{ Metrics_params.param.batch_size}")
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
