import numpy as np
from astropy.table import Table

from btk.metrics import get_detection_match


def test_true_detected_catalog():
    """Test if correct matches are computed from the true and detected tables"""
    names = ["x_peak", "y_peak"]
    cols = [[0.0, 1.0], [0.0, 0.0]]
    true_table = Table(cols, names=names)
    detected_table = Table([[0.1], [0.1]], names=["x_peak", "y_peak"])
    matches = get_detection_match(true_table, detected_table)
    target_num_detections = np.array([0, -1])
    np.testing.assert_array_equal(
        matches["match_detected_id"],
        target_num_detections,
        err_msg="Incorrect match",
    )
    np.testing.assert_almost_equal(
        matches["dist"],
        [0.14142135623730953, 0.0],
        decimal=6,
        err_msg="Incorrect distance between true centers",
    )


def test_no_detection():
    """When no detection, make sure no match is returned"""
    names = ["x_peak", "y_peak"]
    cols = [[0.0], [0.0]]
    true_table = Table(cols, names=names)
    detected_table = Table([[], []], names=["x_peak", "y_peak"])
    matches = get_detection_match(true_table, detected_table)
    np.testing.assert_array_equal(
        matches["match_detected_id"], [-1], err_msg="A match was returned when it should not have."
    )
