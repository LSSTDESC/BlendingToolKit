import numpy as np
import pytest
import btk
from astropy.table import Table


def test_true_detected_catalog():
    """Test if correct entries are computed for the true and detected table.

    The columns of true table that are tested are:
    number of detections num_detections1 and num_detections2, closest detection
    ids: closest_det_id1 and closest_det_id2, and minimum distance between true
    centers: min_dist.

    The columns of detected table that are tested are:
    match ids: match_true_id1 and match_true_id1, match_galtileid1,
    match_galtileid2, minimum distances: d and dSigma.

    """
    names = ['dx', 'dy', 'size', 'galtileid', 'i_ab', 'redshift']
    cols = [[0., 1.], [0., 0.], [2., 0.5], [1, 2],
            [20, 21], [0.1, 0.2]]
    true_tables = [Table(cols, names=names)]
    detected_tables = [Table([[0.1], [0.1]], names=['dx', 'dy'])]
    # det is list of true catalog, detected catalog, detection summary
    det = btk.compute_metrics.evaluate_detection(
        true_tables, detected_tables, 0)
    # Test true catalog
    target_num_detections = np.array([1, 0])
    np.testing.assert_array_equal(
        det[0]['num_detections1'], target_num_detections,
        err_msg="Incorrect detection algorithm 1")
    np.testing.assert_array_equal(
        det[0]['num_detections2'], target_num_detections,
        err_msg="Incorrect detection algorithm 2")
    np.testing.assert_array_equal(
        det[0]['min_dist'], [1., 1.],
        err_msg="Incorrect distance between true centers")
    target_detection = np.array([0, 0])
    np.testing.assert_array_equal(
        det[0]['closest_det_id1'], target_detection,
        err_msg="Incorrect closest detection algorithm 1")
    np.testing.assert_array_equal(
        det[0]['closest_det_id2'], target_detection,
        err_msg="Incorrect closest detection algorithm 2")
    np.testing.assert_array_equal(
        det[0]['min_dist'], [1., 1.],
        err_msg="Incorrect distance between true centers")
    np.testing.assert_array_equal(
        det[0]['dm_match'], [0, 1], err_msg="Incorrect dm_match")
    np.testing.assert_array_equal(
        det[0]['dz_match'], [0, 0.1], err_msg="Incorrect dz_match")
    # Test detection catalog
    np.testing.assert_array_equal(
        det[1]['match_true_id1'], [0], err_msg="Incorrect match algorithm 1")
    np.testing.assert_array_equal(
        det[1]['match_true_id2'], [0], err_msg="Incorrect match algorithm 2")
    np.testing.assert_array_equal(
        det[1]['match_galtileid1'], [1], err_msg="Incorrect match algorithm 1")
    np.testing.assert_array_equal(
        det[1]['match_galtileid2'], [1], err_msg="Incorrect match algorithm 2")
    np.testing.assert_array_equal(
        det[1]['dSigma_min'], [0.07071067811865477],
        err_msg="Incorrect min dSigma algorithm 1")
    np.testing.assert_array_equal(
        det[1]['d_min'], [0.14142135623730953],
        err_msg="Incorrect min d algorithm 2")


def no_detection():
    """When no detection, make sure no match is returned"""
    names = ['dx', 'dy', 'size', 'galtileid', 'i_ab', 'redshift']
    cols = [[0.], [0.], [1.], [0.], [20], [0.1]]
    true_tables = [Table(cols, names=names)]
    detected_tables = [Table([[], []], names=['dx', 'dy'])]
    det = btk.compute_metrics.evaluate_detection(
        true_tables, detected_tables, 0)
    target_summary = np.array([[1, 0, 1, 0, 0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(det[2], target_summary,
                                  err_msg="Incorrect detection summary")
    np.testing.assert_array_almost_equal(
        det[0]['dm_match'], [0], err_msg="Incorrect dm_match")
    np.testing.assert_array_almost_equal(
        det[0]['dz_match'], [0], err_msg="Incorrect dz_match")


def one_detection():
    """When one detection, make sure correct match is returned"""
    names = ['dx', 'dy', 'size', 'galtileid', 'i_ab', 'redshift']
    cols = [[0.], [0.], [1.], [0.], [20], [0.1]]
    true_tables = [Table(cols, names=names)]
    detected_tables = [Table([[0.1], [0.1]], names=['dx', 'dy'])]
    det = btk.compute_metrics.evaluate_detection(
        true_tables, detected_tables, 0)
    target_summary = np.array([[1, 1, 0, 0, 0, 1, 0, 0, 0]])
    np.testing.assert_array_equal(
        det[2], target_summary,
        err_msg="Incorrect detection summary")
    np.testing.assert_array_almost_equal(
        det[0]['dm_match'], [0], err_msg="Incorrect dm_match")
    np.testing.assert_array_almost_equal(
        det[0]['dz_match'], [0], err_msg="Incorrect dz_match")


def one_undetected():
    """When one detection, if detection is equidistant to two true centers, but
    second object is larger, then algorithm 1 returns match with object 2 while
    algorithm 2 returns match with object 1"""
    names = ['dx', 'dy', 'size', 'galtileid', 'i_ab', 'redshift']
    cols = [[-1., 1.], [0., 0.], [1., 2.], [100, 101],
            [21, 21.5], [0.1, 0.3]]
    true_tables = [Table(cols, names=names)]
    detected_tables = [Table([[0.], [0.]], names=['dx', 'dy'])]
    det = btk.compute_metrics.evaluate_detection(
        true_tables, detected_tables, 0)
    # Test true catalog
    np.testing.assert_array_equal(
        det[0]['num_detections1'], [1, 0],
        err_msg="Incorrect detection algorithm 1")
    np.testing.assert_array_equal(
        det[0]['num_detections2'], [0, 1],
        err_msg="Incorrect detection algorithm 2")
    np.testing.assert_array_equal(
        det[0]['closest_det_id1'], [0, 0],
        err_msg="Incorrect closest detection algorithm 1")
    np.testing.assert_array_equal(
        det[0]['closest_det_id2'], [0, 0],
        err_msg="Incorrect closest detection algorithm 2")
    np.testing.assert_array_equal(
        det[0]['min_dist'], [2., 2.],
        err_msg="Incorrect distance between true centers")
    np.testing.assert_array_almost_equal(
        det[0]['dm_match'], [0, 0.5], err_msg="Incorrect dm_match")
    np.testing.assert_array_almost_equal(
        det[0]['dz_match'], [0, 0.2], err_msg="Incorrect dm_match")
    # Test detection catalog
    np.testing.assert_array_equal(
        det[1]['match_true_id1'], [0],
        err_msg="Incorrect match algorithm in 1")
    np.testing.assert_array_equal(
        det[1]['match_true_id2'], [1],
        err_msg="Incorrect match algorithm in 2")
    np.testing.assert_array_equal(
        det[1]['match_galtileid1'], [100],
        err_msg="Incorrect match algorithm 1")
    np.testing.assert_array_equal(
        det[1]['match_galtileid2'], [101],
        err_msg="Incorrect match algorithm 2")
    np.testing.assert_array_equal(
        det[1]['dSigma_min'], [0.5],
        err_msg="Incorrect min dSigma algorithm 1")
    np.testing.assert_array_equal(
        det[1]['d_min'], [1.], err_msg="Incorrect minimum match distance")
    target_summary = np.array([[2, 1, 1, 0, 0, 1, 1, 0, 0]])
    np.testing.assert_array_equal(det[2], target_summary,
                                  err_msg="Incorrect detection summary")


def test_m_z_diff():
    """Test is correct dz_match and dr_match values are computed in
    compute_metrics.get_m_z_diff. One detection has no match."""
    names = ['dx', 'dy', 'size', 'galtileid', 'i_ab', 'redshift']
    cols = [[-6., 0., 1.], [-6., 0., 1], [1., 2., 1.], [100, 101, 102],
            [21, 21.5, 23], [0.1, 0.3, 0.05]]
    true_tables = [Table(cols, names=names)]
    detected_tables = [Table([[0.1, 1.1], [0.1, 1.1]], names=['dx', 'dy'])]
    det = btk.compute_metrics.evaluate_detection(
        true_tables, detected_tables, 0)
    # Test true catalog
    np.testing.assert_array_equal(
        det[0]['num_detections1'], [0, 1, 1],
        err_msg="Incorrect detection algorithm 1")
    np.testing.assert_array_equal(
        det[1]['match_true_id1'], [1, 2],
        err_msg="Incorrect closest detection algorithm 1")
    np.testing.assert_array_equal(
        det[0]['closest_det_id1'], [0, 0, 1],
        err_msg="Incorrect closest detection algorithm 1")
    np.testing.assert_array_almost_equal(
        det[0]['ddist_match'], [8.485281374238571, 0, 0],
        err_msg="Incorrect distance between true centers")
    np.testing.assert_array_almost_equal(
        det[0]['dm_match'], [-0.5, 0, 0], err_msg="Incorrect dm_match")
    np.testing.assert_array_almost_equal(
        det[0]['dz_match'], [-0.2, 0, 0], err_msg="Incorrect dm_match")


@pytest.mark.timeout(5)
def test_detection():
    no_detection()
    one_detection()
    one_undetected()
