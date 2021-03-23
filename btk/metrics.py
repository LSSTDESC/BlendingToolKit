from abc import ABC
from abc import abstractmethod

import astropy.table
import numpy as np
import scipy.spatial


class MetricsParams(ABC):
    def __init__(self, meas_generator, batch_size):
        """Class describing functions to return results of
        detection/deblending/measurement algorithm in meas_generator. Each
        blend results yielded by the meas_generator for a batch.
        """
        self.meas_generator = meas_generator
        self.batch_size = batch_size

    @abstractmethod
    def get_detections(self):
        """
        Returns detection results as two catalogs one with entries of true
        objects in blend and the other with the detection.

        Overwrite this function to return results from the detection algorithm.

        Returns:
            list:  List of astropy tables with the blend catalogs used to draw
                the blend scene in the batch. Length of tables must be equal to
                the batch size. x and y coordinate values must be under columns
                named 'x_peak' and 'y_peak' respectively, in pixels from bottom left
                corner as (0, 0).
            list: List of astropy Tables of output with the outputs of the
            detection algorithm. Length of tables must be equal to the batch
            size. x and y coordinate values must be under columns named 'x_peak'
            and 'y_peak' respectively, in pixels from bottom left corner as (0, 0).
        """
        pass

    def get_segmentation(self):
        """Define function here to return results from the segmentation
        algorithm.
        """
        pass

    def get_flux(self):
        """Define function here to return results from the flux measurement
        algorithm.
        """
        pass

    def get_shapes(self):
        """Define function here to return results from the shape measurement
        algorithm.
        """
        pass


class BasicMetricsParams(MetricsParams):
    """Class describing functions to return results of
    detection/deblending/measurement algorithm in meas_generator. Each
    time the algorithm is called, it is run on a batch of blends yielded
    by the meas_generator.
    """

    def get_detections(self):
        """Returns input blend catalog and detection catalog for
        the detection performed.

        Returns:
            Results of the detection algorithm are returned as:
                true_tables: List of astropy Table of the blend catalogs of the
                    batch. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'x_peak' and 'y_peak'
                    respectively, in pixels from bottom left corner as (0, 0).
                detected_tables: List of astropy Table of output from detection
                    algorithm. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'x_peak' and 'y_peak'
                    respectively, in pixels from bottom left corner as (0, 0).
        """
        blend_op, deblend_op, _ = next(self.meas_generator)
        true_tables = blend_op["blend_list"]
        detected_tables = []
        for i in range(len(true_tables)):
            detected_centers = deblend_op[i]["peaks"]
            detected_table = astropy.table.Table(detected_centers, names=["x_peak", "y_peak"])
            detected_tables.append(detected_table)
        return true_tables, detected_tables


def make_true_seg_map(image, threshold):
    """Returns a boolean segmentation map corresponding to pixels in
    image above a certain threshold value.
    Args:
        image: Image to estimate segmentation map of
        threshold: Pixels above this threshold are marked as belonging to
            segmentation map

    Returns:
        Boolean segmentation map of the image
    """
    seg_map = np.zeros_like(image)
    seg_map[image < threshold] = 0
    seg_map[image >= threshold] = 1
    return seg_map.astype(np.bool)


def get_closest_neighbor_distance(true_table):
    """Returns a astropy.table.column with the distance to the closest object.

    Function uses scipy.spatial to compute distance between the object centers.
    If object is the only one in the blend then the `min_dist` value is set to
    np.inf.

    Args:
        true_table: Catalog with entries corresponding to one blend.

    Returns:
        `astropy.table.Column`s: size of the galaxy.
    """
    peaks = np.stack([np.array(true_table["x_peak"]), np.array(true_table["y_peak"])]).T
    if peaks.shape[0] > 1:
        distance = scipy.spatial.distance.cdist(peaks, peaks, metric="euclidean")
        min_dist = [np.min(distance[i][distance[i] > 0]) for i in range(len(distance))]
        true_table["min_dist"] = min_dist


def get_m_z_diff(true_table, detected_true):
    """Updates the input astropy.table.column with the difference in magnitude,
    and redshift between an object and it's algorithm matches. It also computes
    the true distance between an object and its closest detection.

    Args:
        detected_true:
        true_table: Catalog with entries corresponding to one blend.
    """
    if len(detected_true) == 0 or len(true_table) == 0:
        # No match since either no true or no matched true objects
        return
    det_centers = np.stack([np.array(detected_true["x_peak"]), np.array(detected_true["y_peak"])]).T
    z_tree = scipy.spatial.KDTree(det_centers)
    true_centers = np.stack([np.array(true_table["x_peak"]), np.array(true_table["y_peak"])]).T
    match = detected_true[z_tree.query(true_centers)[1]]
    true_table["dm_match"] = true_table["i_ab"] - match["i_ab"]
    true_table["dz_match"] = true_table["redshift"] - match["redshift"]
    x_peak = true_table["x_peak"] - match["x_peak"]
    y_peak = true_table["y_peak"] - match["y_peak"]
    true_table["ddist_match"] = np.hypot(x_peak, y_peak)


def initialize_detection_tables(detected_table, true_table, batch_index, batch_size, blend_index):
    """Initialize column entries of true objects and detection catalog to their
    default values.

    This is necessary since, if either true objects or detection table is
    empty, then there is no match in get_detection_match and those column
    entries would not otherwise be present in the tables.

    Function does not return anything, only the astropy tables are updates.
    """
    # initialize true objects table columns
    num_true = len(true_table)
    true_table["true_id"] = range(num_true)  # id in blend [0-num_true]
    # index of blend in test size [0 - len(blend_summary)]
    true_table["blend_index"] = np.ones(num_true, dtype=int) * (
        batch_index * batch_size + blend_index
    )
    # index of batch in test size [0 - batch_size]
    true_table["batch_index"] = np.ones(num_true, dtype=int) * batch_index
    # number of times object was detected [0 - num_det]
    true_table["num_detections1"] = np.zeros(num_true, dtype=int)
    true_table["num_detections2"] = np.zeros(num_true, dtype=int)
    # detection id of closest detection with 2 algorithms [0 - num_det]
    true_table["closest_det_id1"] = np.ones(num_true, dtype=int) * -1.0
    true_table["closest_det_id2"] = np.ones(num_true, dtype=int) * -1.0
    # difference in centroids, i band magnitude and redshift between an object
    # and its match with algorithm 2
    true_table["dm_match"] = np.zeros(num_true, dtype=int)
    true_table["dz_match"] = np.zeros(num_true, dtype=int)
    true_table["ddist_match"] = np.zeros(num_true, dtype=int)
    true_table["dnorm_dist_match"] = np.zeros(num_true, dtype=int)
    # find distance to nearest neighbor. If isolated then set to np.inf
    true_table["min_dist"] = np.ones(num_true) * np.inf
    get_closest_neighbor_distance(true_table)
    # initialize detected objects table columns
    num_det = len(detected_table)
    detected_table["detection_id"] = range(num_det)  # id in blend [0-num_det]
    # index of blend in test size [0 - len(blend_summary)]
    detected_table["blend_index"] = np.ones(num_det, dtype=int) * (
        batch_index * batch_size + blend_index
    )
    # index of batch in test size [0 - batch_size]
    detected_table["batch_index"] = np.ones(num_det, dtype=int) * batch_index
    # id of closest true object; [0 - num_true] if detected, else -1
    detected_table["match_true_id1"] = np.ones(num_det, dtype=int) * -1
    detected_table["match_true_id2"] = np.ones(num_det, dtype=int) * -1
    detected_table["match_galtileid1"] = np.ones(num_det, dtype=int) * -1
    detected_table["match_galtileid2"] = np.ones(num_det, dtype=int) * -1


def get_detection_match(true_table, detected_table):
    """Match detections to true objects and update values in the input
    blend catalog and detection catalog.

    Function does not return anything, only the astropy tables are updated.

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        detected_table(astropy.table.Table): Table with entries corresponding
            to output of measurement algorithm in one blend.
    """
    if len(detected_table) == 0 or len(true_table) == 0:
        # No match since either no detection or no true objects
        return
    t_x = true_table["x_peak"][:, np.newaxis] - detected_table["x_peak"]
    t_y = true_table["y_peak"][:, np.newaxis] - detected_table["y_peak"]
    dist = np.hypot(t_x, t_y)
    norm_size = true_table["size"]
    norm_dist = dist / norm_size[:, np.newaxis]
    detected_table["dSigma_min"] = np.min(norm_dist, axis=0)
    detected_table["d_min"] = np.min(dist, axis=0)
    detection_threshold1 = 5
    condlist1 = [
        np.min(dist, axis=0) <= detection_threshold1,
        np.min(dist, axis=0) > detection_threshold1,
    ]
    choicelist1 = [np.argmin(dist, axis=0), -1]
    match_id1 = np.select(condlist1, choicelist1)
    detected_table["match_true_id1"] = match_id1
    detected_table["match_galtileid1"] = true_table["galtileid"][match_id1]
    detection_threshold2 = 0.5
    condlist2 = [
        np.min(norm_dist, axis=0) <= detection_threshold2,
        np.min(norm_dist, axis=0) > detection_threshold2,
    ]
    choicelist2 = [np.argmin(norm_dist, axis=0), -1]
    match_id2 = np.select(condlist2, choicelist2)
    detected_table["match_true_id2"] = match_id2
    detected_table["match_galtileid2"] = true_table["galtileid"][match_id2]
    np.testing.assert_array_equal(
        np.argmin(dist, axis=1),
        np.argmin(norm_dist, axis=1),
        err_msg="norm_dist computation is wrong.",
    )
    true_table["closest_det_id1"] = np.argmin(norm_dist, axis=1)
    for j in detected_table["match_true_id1"]:
        if j > -1:
            true_table["num_detections1"][j] += 1
    true_table["closest_det_id2"] = np.argmin(dist, axis=1)
    for j in detected_table["match_true_id2"]:
        if j > -1:
            true_table["num_detections2"][j] += 1
    get_m_z_diff(true_table, true_table[match_id1[match_id1 >= 0]])


def get_blend_detection_summary(true_table, det_table):
    """Returns list summarizing results of detection metric computation

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        det_table(astropy.table.Table): Table with entries corresponding
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
    num_detected1 = len(np.where(true_table["num_detections1"] == 1)[0])
    num_undetected1 = len(np.where(true_table["num_detections1"] == 0)[0])
    num_spurious1 = len(np.where(det_table["match_true_id1"] == -1)[0])
    num_shred1 = len(np.where(true_table["num_detections1"] > 1)[0])
    num_detected2 = len(np.where(true_table["num_detections2"] == 1)[0])
    num_undetected2 = len(np.where(true_table["num_detections2"] == 0)[0])
    num_spurious2 = len(np.where(det_table["match_true_id2"] == -1)[0])
    num_shred2 = len(np.where(true_table["num_detections2"] > 1)[0])
    if not num_detected1 + num_undetected1 + num_shred1 == num_true:
        raise ValueError(
            "Number of detected objects + number undetected "
            "objects must be equal to the total number of true "
            "objects"
        )

    if not num_detected2 + num_undetected2 + num_shred2 == num_true:
        raise ValueError(
            "Number of detected objects + number undetected "
            "objects must be equal to the total number "
            "of true objects."
        )

    num_matched_detections1 = true_table["num_detections1"].sum()
    if not num_matched_detections1 + num_spurious1 == num_det:
        raise ValueError(
            "Number of detections match to a true object + "
            "number of spurious must be equal to the "
            "total number of detections."
        )

    num_matched_detections2 = true_table["num_detections2"].sum()
    if not num_matched_detections2 + num_spurious2 == num_det:
        raise ValueError(
            "Number of detections match to a true object + "
            "number of spurious must be equal to the total "
            "number of detections."
        )

    blend_summary = [
        num_true,
        num_detected1,
        num_undetected1,
        num_spurious1,
        num_shred1,
        num_detected2,
        num_undetected2,
        num_spurious2,
        num_shred2,
    ]
    return blend_summary


def get_detection_eff_matrix(summary_table, num):
    """Computes the detection efficiency matrix for the input detection summary
    table.

    Input argument num sets the maximum number of true objects per blend in the
    test set for which the
    detection efficiency matrix is to be created for. Detection efficiency is
    computed for a number of true objects in the range (0-num) as columns and
    the detection percentage as rows. The percentage values in a column sum to
    100.

    The input summary table must be a numpy array of shape [N, 5], where N is
    the test set size. The 5 columns in the summary_table are number of true
    objects, detected sources, undetected objects, spurious detections and
    shredded objects for each of the N blend scenes in the test set.

    Args:
        summary_table (`numpy.array`): Detection summary as a table [N, 5].
        num (int): Maximum number of true objects to create matrix for. Number
            of columns in efficiency matrix will be num+1. The first column
            will correspond to no true objects.

    Returns:
        numpy.ndarray of size[num+2, num+1] that shows detection efficiency.
    """
    eff_matrix = np.zeros((num + 2, num + 1))
    for i in range(0, num + 1):
        (q_true,) = np.where(summary_table[:, 0] == i)
        for j in range(0, num + 2):
            if len(q_true) > 0:
                (q_det,) = np.where(summary_table[q_true, 1] == j)
                eff_matrix[j, i] = len(q_det)
    norm = np.sum(eff_matrix, axis=0)
    # If no detections along a column, set sum to 1 to avoid dividing by zero.
    norm[norm == 0.0] = 1
    # normalize over columns.
    eff_matrix = eff_matrix / norm[np.newaxis, :] * 100.0
    return eff_matrix


def evaluate_detection(true_tables, detected_tables, batch_index):
    """
    Compares the true centers and detected centers to identify the
    number of true detections, number of sources that were undetected
    and number of spurious detections.
    Args:
        true_tables:  List of astropy Tables of the blend catalogs of the
            batch. Length of tables must be the batch size. x and y coordinate
            values must be under columns named 'x_peak' and 'y_peak' respectively, in
            pixels from bottom left corner as (0, 0).
        detected_tables: List of astropy Tables of output from detection
            algorithm. Length of tables must be the batch size. x and y
            coordinate values must be under columns named 'x_peak' and 'y_peak'
            respectively, in pixels from bottom left corner as (0, 0).
        batch_index(int): Index number of the batch.
    Returns:
        batch_true_table: astropy.table.Table with parameters of true galaxy in
            the batch.
        batch_detected_table: astropy.table.Table with parameters of detected
            objects in the batch.
        batch_blend_list: List summarizing detection match results.
    """
    batch_true_table = astropy.table.Table()
    batch_detected_table = astropy.table.Table()
    batch_size = len(true_tables)
    batch_blend_summary = []
    for i in range(batch_size):
        true_table = true_tables[i]
        detected_table = detected_tables[i]
        # initialize columns to default values
        initialize_detection_tables(detected_table, true_table, batch_index, batch_size, i)
        # match detection and true source
        get_detection_match(true_table, detected_table)
        # summarize blend detection results to table
        blend_summary = get_blend_detection_summary(true_table, detected_table)
        batch_blend_summary.append(blend_summary)
        # add results to batch table
        batch_detected_table = astropy.table.vstack((batch_detected_table, detected_table))
        batch_true_table = astropy.table.vstack((batch_true_table, true_table))
    return batch_true_table, batch_detected_table, batch_blend_summary


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


def run(metrics_params, test_size=1000):
    """Runs detection/segmentation/flux/shape measurement algorithm defined in
    the input metrics params for input test_size number of btk runs.

    Args:
        metrics_params: Instance from class
        `btk.metrics.Metrics_params` describing functions to return
        results of detection/deblending/measurement algorithm.
        test_size(int): Number of times Metrics_params is run and results
            summarized.

    Returns:
        dict summarizing detection/deblending/measurement results.

    """
    results = {
        "detection": [astropy.table.Table(), astropy.table.Table(), []],
        "segmentation": [],
        "flux": [],
        "shapes": [],
    }
    for i in range(test_size):
        print(f"Running test {i}")
        # Evaluate detection algorithm
        try:
            batch_detection_result = metrics_params.get_detections()
        except GeneratorExit as e:
            print(e)
            print("GeneratorExit encountered. Returning results")
            return results
        if (
            len(batch_detection_result[0]) != len(batch_detection_result[1])
            or len(batch_detection_result[0]) != metrics_params.batch_size
        ):
            raise ValueError(
                "Metrics_params.get_detections output must be "
                "two lists of astropy table of length batch size."
                f" Found {len(batch_detection_result[0])}, "
                f"{len(batch_detection_result[1])}, "
                f"{metrics_params.batch_size}"
            )
        true_table, detected_table, detection_summary = evaluate_detection(
            batch_detection_result[0], batch_detection_result[1], batch_index=i
        )
        results["detection"][0] = astropy.table.vstack([results["detection"][0], true_table])
        results["detection"][1] = astropy.table.vstack([results["detection"][1], detected_table])
        results["detection"][2].extend(detection_summary)
        # Evaluate segmentation algorithm
        segmentation = metrics_params.get_segmentation()
        results["segmentation"].append(evaluate_segmentation(segmentation, index=i))
        # Evaluate flux measurement algorithm
        flux = metrics_params.get_flux()
        results["flux"].append(evaluate_flux(flux, index=i))
        # Evaluate shape measurement algorithm
        shapes = metrics_params.get_shapes()
        results["shapes"].append(evaluate_shapes(shapes, index=i))
    return results
