import numpy as np
import btk.metrics


def test_detection_eff_matrix():
    """Tests detection efficiency matrix computation in utils by inputting a
    summary table with 4 entries, with number of true sources between 1-4 and
    all detected and expecting matrix with
    secondary diagonal being one"""
    summary = np.array(
        [[1, 1, 0, 0, 0], [2, 2, 0, 0, 0], [3, 3, 0, 0, 0], [4, 4, 0, 0, 0]]
    )
    num = 4
    eff_matrix = btk.metrics.get_detection_eff_matrix(summary, num)
    test_eff_matrix = np.eye(num + 2)[:, : num + 1] * 100
    test_eff_matrix[0, 0] = 0.0
    np.testing.assert_array_equal(
        eff_matrix, test_eff_matrix, err_msg="Incorrect efficiency matrix"
    )
    pass
