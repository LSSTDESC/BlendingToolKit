from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import btk.metrics


def get_metrics_generator(meas_function, cpus=1, measure_kwargs=None):
    """Returns draw generator with group sampling function"""

    np.random.seed(0)
    catalog_name = "data/sample_input_catalog.fits"
    stamp_size = 24
    survey = btk.survey.Rubin
    shifts = [
        [[-0.3, 1.2], [-1.6, -1.7]],
        [[-1.1, -2.1], [1.4, 1.8]],
        [[-1.8, -0.8], [-0.6, 2.2]],
        [[-2.0, -0.7], [-2.2, 1.9]],
        [[1.1, -1.5], [0.1, -2.3]],
        [[-2.3, 1.9], [0.4, -1.9]],
        [[2.0, -2.0], [2.0, 0.1]],
        [[0.2, 2.4], [-1.8, -2.0]],
    ]
    indexes = [[4, 5], [9, 1], [9, 2], [0, 2], [3, 8], [0, 7], [10, 2], [0, 10]]
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    draw_blend_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        btk.sampling_functions.DefaultSampling(),
        [survey],
        shifts=shifts,
        indexes=indexes,
        stamp_size=stamp_size,
    )
    meas_generator = btk.measure.MeasureGenerator(
        meas_function, draw_blend_generator, cpus=cpus, measure_kwargs=measure_kwargs
    )
    metrics_generator = btk.metrics.MetricsGenerator(
        meas_generator,
        use_metrics=("detection", "segmentation", "reconstruction"),
        target_meas={"ellipticity": btk.metrics.meas_ksb_ellipticity},
    )
    return metrics_generator


@patch("btk.plot_utils.plt.show")
def test_sep_metrics(mock_show):
    metrics_generator = get_metrics_generator(btk.measure.sep_measure)
    _, _, results = next(metrics_generator)
    results = list(results.values())[0]
    gal_summary = results["galaxy_summary"][
        results["galaxy_summary"]["detected"] == True  # noqa: E712
    ]
    msr = gal_summary["msr"]
    dist = gal_summary["distance_closest_galaxy"]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    btk.plot_utils.plot_metrics_distribution(msr, "msr", ax1, upper_quantile=0.9)
    btk.plot_utils.plot_metrics_correlation(
        dist, msr, "Distance to the closest galaxy", "msr", ax2, upper_quantile=0.9, style="heatmap"
    )
    btk.plot_utils.plot_efficiency_matrix(results["detection"]["eff_matrix"])
    plt.close("all")


def test_detection_eff_matrix():
    """Tests detection efficiency matrix computation in utils by inputting a
    summary table with 4 entries, with number of true sources between 1-4 and
    all detected and expecting matrix with
    secondary diagonal being one"""
    summary = np.array([[1, 1, 0, 0, 0], [2, 2, 0, 0, 0], [3, 3, 0, 0, 0], [4, 4, 0, 0, 0]])
    num = 4
    eff_matrix = btk.metrics.get_detection_eff_matrix(summary, num)
    test_eff_matrix = np.eye(num + 2)[:, : num + 1] * 100
    test_eff_matrix[0, 0] = 0.0
    np.testing.assert_array_equal(
        eff_matrix, test_eff_matrix, err_msg="Incorrect efficiency matrix"
    )


@patch("btk.plot_utils.plt.show")
def test_measure_kwargs(mock_show):
    """Test detection with sep"""
    meas_generator = get_metrics_generator(
        btk.measure.sep_measure, measure_kwargs=[{"sigma_noise": 2.0}, {"sigma_noise": 3.0}]
    )
    _, _, results = next(meas_generator)
    average_precision = btk.metrics.auc(results, "sep_measure", 2, plot=True)
    assert average_precision == 0.4375
