from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from conftest import data_dir

import btk.plot_utils as plot_utils
from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.measure import MeasureGenerator, sep_singleband_measure
from btk.metrics import (
    MetricsGenerator,
    auc,
    get_detection_eff_matrix,
    meas_ksb_ellipticity,
)
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys

TEST_SEED = 0


def get_metrics_generator(
    meas_function,
    cpus=1,
    measure_kwargs=None,
):
    """Returns draw generator with group sampling function"""
    catalog_name = data_dir / "sample_input_catalog.fits"
    stamp_size = 24
    survey = get_surveys("LSST")
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
    catalog = CatsimCatalog.from_file(catalog_name)
    draw_blend_generator = CatsimGenerator(
        catalog,
        DefaultSampling(seed=TEST_SEED),
        [survey],
        shifts=shifts,
        indexes=indexes,
        stamp_size=stamp_size,
        seed=TEST_SEED,
    )
    meas_generator = MeasureGenerator(
        meas_function, draw_blend_generator, cpus=cpus, measure_kwargs=measure_kwargs
    )
    metrics_generator = MetricsGenerator(
        meas_generator,
        target_meas={"ellipticity": meas_ksb_ellipticity},
    )
    return metrics_generator


@patch("btk.plot_utils.plt.show")
def test_sep_metrics(mock_show):
    metrics_generator = get_metrics_generator(sep_singleband_measure)
    blend_results, meas_results, metrics_results = next(metrics_generator)
    gal_summary = metrics_results["galaxy_summary"]["sep_singleband_measure"]
    gal_summary = gal_summary[gal_summary["detected"] == True]  # noqa: E712
    msr = gal_summary["msr"]
    dist = gal_summary["distance_closest_galaxy"]
    _, (ax1, ax2) = plt.subplots(1, 2)
    plot_utils.plot_metrics_distribution(msr, "msr", ax1, upper_quantile=0.9)
    plot_utils.plot_metrics_correlation(
        dist, msr, "Distance to the closest galaxy", "msr", ax2, upper_quantile=0.9, style="heatmap"
    )
    plot_utils.plot_metrics_summary(
        metrics_results,
        target_meas_keys=["ellipticity0"],
        target_meas_limits=[[-1, 1]],
        interactive=False,
    )
    plot_utils.plot_metrics_summary(
        metrics_results,
        target_meas_keys=["ellipticity0"],
        target_meas_limits=[[-1, 1]],
        interactive=True,
    )
    plot_utils.plot_with_deblended(
        blend_results["blend_images"],
        blend_results["isolated_images"],
        blend_results["blend_list"],
        meas_results["catalog"]["sep_singleband_measure"],
        meas_results["deblended_images"]["sep_singleband_measure"],
        metrics_results["matches"]["sep_singleband_measure"],
        indexes=list(range(5)),
        band_indices=[1, 2, 3],
    )
    plt.close("all")


@patch("btk.plot_utils.plt.show")
def test_measure_kwargs(mock_show):
    """Test detection with sep"""
    metrics_generator = get_metrics_generator(
        sep_singleband_measure, measure_kwargs=[{"sigma_noise": 2.0}, {"sigma_noise": 3.0}]
    )
    _, _, results = next(metrics_generator)
    average_precision = auc(results, "sep_singleband_measure", 2, plot=True)
    assert average_precision == 0.25


def test_detection_eff_matrix():
    """Tests detection efficiency matrix computation in utils by inputting a
    summary table with 4 entries, with number of true sources between 1-4 and
    all detected and expecting matrix with
    secondary diagonal being one"""
    summary = np.array([[1, 1, 0, 0, 0], [2, 2, 0, 0, 0], [3, 3, 0, 0, 0], [4, 4, 0, 0, 0]])
    num = 4
    eff_matrix = get_detection_eff_matrix(summary, num)
    test_eff_matrix = np.eye(num + 2)[:, : num + 1] * 100
    test_eff_matrix[0, 0] = 0.0
    np.testing.assert_array_equal(
        eff_matrix, test_eff_matrix, err_msg="Incorrect efficiency matrix"
    )
