import numpy as np
from conftest import data_dir

from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.measure import basic_measure
from btk.measure import MeasureGenerator
from btk.measure import sep_measure
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys

TEST_SEED = 0


def get_meas_results(meas_function, cpus=1, measure_kwargs=None):
    """Returns draw generator with group sampling function"""

    catalog_name = data_dir / "sample_input_catalog.fits"
    stamp_size = 24
    survey = get_surveys("LSST")
    shifts = [[-0.3, 1.2]]
    indexes = [[1]]
    catalog = CatsimCatalog.from_file(catalog_name)
    draw_blend_generator = CatsimGenerator(
        catalog,
        DefaultSampling(seed=TEST_SEED),
        [survey],
        shifts=shifts,
        indexes=indexes,
        stamp_size=stamp_size,
        batch_size=1,
        seed=TEST_SEED,
    )
    meas_generator = MeasureGenerator(
        meas_function, draw_blend_generator, cpus=cpus, measure_kwargs=measure_kwargs
    )
    blend_results, results = next(meas_generator)
    wcs = blend_results["wcs"]
    x, y = wcs.world_to_pixel_values(shifts[0][0] / 3600, shifts[0][1] / 3600)
    target = np.array([x.item(), y.item()])
    return target, results


def compare_sep():
    """Test detection with sep"""
    target, results = get_meas_results(sep_measure, measure_kwargs=[{"sigma_noise": 2.0}])
    x_peak, y_peak = (
        results["catalog"]["sep_measure"][0]["x_peak"].item(),
        results["catalog"]["sep_measure"][0]["y_peak"].item(),
    )
    detected_centers = np.array([x_peak, y_peak])
    dist = np.max(np.abs(detected_centers - target))
    np.testing.assert_array_less(dist, 1.0)


def compare_sep_multiprocessing():
    """Test detection with sep"""
    target, results = get_meas_results(sep_measure, measure_kwargs=[{"sigma_noise": 2.0}], cpus=4)
    x_peak, y_peak = (
        results["catalog"]["sep_measure"][0]["x_peak"].item(),
        results["catalog"]["sep_measure"][0]["y_peak"].item(),
    )
    detected_centers = np.array([x_peak, y_peak])
    dist = np.max(np.abs(detected_centers - target))
    np.testing.assert_array_less(dist, 1.0)


def test_algorithms():
    """Test detection/deblending/measurement algorithms if installed"""
    compare_sep()
    compare_sep_multiprocessing()
    get_meas_results(basic_measure, cpus=4)


def test_measure_kwargs():
    """Test detection with sep"""
    target, results = get_meas_results(
        sep_measure, measure_kwargs=[{"sigma_noise": 2.0}, {"sigma_noise": 3.0}]
    )
    x_peak, y_peak = (
        results["catalog"]["sep_measure0"][0]["x_peak"].item(),
        results["catalog"]["sep_measure0"][0]["y_peak"].item(),
    )
    detected_centers = np.array([x_peak, y_peak])
    dist = np.max(np.abs(detected_centers - target))
    np.testing.assert_array_less(dist, 1.0)
