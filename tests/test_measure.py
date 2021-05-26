import numpy as np
from conftest import data_dir

import btk.catalog
import btk.measure
import btk.sampling_functions
import btk.survey


def get_meas_generator(meas_function, cpus=1, measure_kwargs=None):
    """Returns draw generator with group sampling function"""

    np.random.seed(0)
    catalog_name = data_dir / "sample_input_catalog.fits"
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
    return meas_generator


def compare_sep():
    """Test detection with sep"""
    meas_generator = get_meas_generator(btk.measure.sep_measure)
    _, results = next(meas_generator)
    x_peak, y_peak = (
        results["catalog"]["sep_measure"][0]["x_peak"].item(),
        results["catalog"]["sep_measure"][0]["y_peak"].item(),
    )
    detected_centers = np.array([x_peak, y_peak])
    target_detection = np.array([65.495, 51.012])
    np.testing.assert_array_almost_equal(
        detected_centers,
        target_detection,
        decimal=3,
        err_msg="Did not get desired detections",
    )


def compare_sep_multiprocessing():
    """Test detection with sep"""
    meas_generator = get_meas_generator(btk.measure.sep_measure, cpus=4)
    _, results = next(meas_generator)
    x_peak, y_peak = (
        results["catalog"]["sep_measure"][0]["x_peak"].item(),
        results["catalog"]["sep_measure"][0]["y_peak"].item(),
    )
    detected_centers = np.array([x_peak, y_peak])
    target_detection = np.array([65.495, 51.012])
    np.testing.assert_array_almost_equal(
        detected_centers,
        target_detection,
        decimal=3,
        err_msg="Did not get desired detections",
    )


def test_algorithms():
    """Test detection/deblending/measurement algorithms if installed"""
    compare_sep()
    compare_sep_multiprocessing()


def test_measure_kwargs():
    """Test detection with sep"""
    meas_generator = get_meas_generator(
        btk.measure.sep_measure, measure_kwargs=[{"sigma_noise": 2.0}, {"sigma_noise": 3.0}]
    )
    _, results = next(meas_generator)
    print(results["catalog"])
    x_peak, y_peak = (
        results["catalog"]["sep_measure0"][0]["x_peak"].item(),
        results["catalog"]["sep_measure0"][0]["y_peak"].item(),
    )
    detected_centers = np.array([x_peak, y_peak])
    target_detection = np.array([65.603, 51.104])
    np.testing.assert_array_almost_equal(
        detected_centers,
        target_detection,
        decimal=3,
        err_msg="Did not get desired detections",
    )
