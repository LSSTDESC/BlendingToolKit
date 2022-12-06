import numpy as np
from conftest import data_dir

from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.measure import (
    MeasureGenerator,
    basic_measure,
    sep_multiband_measure,
    sep_singleband_measure,
)
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys

TEST_SEED = 0


def get_meas_results(meas_function, cpus=1, measure_kwargs=None):
    """Runs a measurement function on a set of data -- returns targets and predicitons"""
    catalog_name = data_dir / "sample_input_catalog.fits"
    stamp_size = 24
    survey = get_surveys("LSST")
    shifts = [
        [-0.3, 1.2],
        [-1.6, -1.7],
        [-1.1, -2.1],
        [1.4, 1.8],
        [-1.8, -0.8],
        [-0.6, 2.2],
        [-2.0, -0.7],
        [-2.2, 1.9],
        [1.1, -1.5],
        [0.1, -2.3],
        [-2.3, 1.9],
        [0.4, -1.9],
        [2.0, -2.0],
        [2.0, 0.1],
        [0.2, 2.4],
        [-1.8, -2.0],
    ]
    indexes = [[4], [5], [9], [1], [9], [2], [0], [2], [3], [8], [0], [7], [10], [2], [0], [10]]
    catalog = CatsimCatalog.from_file(catalog_name)
    draw_blend_generator = CatsimGenerator(
        catalog,
        DefaultSampling(seed=TEST_SEED),
        [survey],
        shifts=shifts,
        indexes=indexes,
        stamp_size=stamp_size,
        seed=TEST_SEED,
        batch_size=8,
    )
    meas_generator = MeasureGenerator(
        meas_function, draw_blend_generator, cpus=cpus, measure_kwargs=measure_kwargs
    )
    blend_results, results = next(meas_generator)
    target = np.array(
        [[blend["x_peak"].item(), blend["y_peak"].item()] for blend in blend_results["blend_list"]]
    )
    return target, results


def compare_sep(cpus=1):
    """
    Test sep detection using single band and multiband

    Runs sep_singleband_measure and sep_multiband_measure on a batch of images, verifying
    that detections are close to the ground truth and number of detections for sep_multiband_measure
    is bigger than sep_singleband_measure.
    """

    target, results = get_meas_results(
        [sep_singleband_measure, sep_multiband_measure],
        measure_kwargs=[{"sigma_noise": 1.5}],
        cpus=cpus,
    )
    # count the number of detected sources for both algroitms
    detected_sources = {"sep_singleband_measure": 0, "sep_multiband_measure": 0}
    for meas_function in detected_sources.keys():
        for i, blend in enumerate(results["catalog"][meas_function]):
            if len(blend) > 0:
                detected_centers = np.array([blend[0]["x_peak"].item(), blend[0]["y_peak"].item()])
                dist = np.max(np.abs(detected_centers - target[i]))
                # make sure that detections are within 1.5 arcsec from truth
                np.testing.assert_array_less(dist, 1.5)
                detected_sources[meas_function] += 1

    assert detected_sources["sep_multiband_measure"] >= 0.5 * len(target)
    assert detected_sources["sep_singleband_measure"] >= 0.5 * len(target)
    assert detected_sources["sep_multiband_measure"] > detected_sources["sep_singleband_measure"]


def test_algorithms():
    """Test detection/deblending/measurement algorithms if installed"""
    compare_sep(cpus=1)
    compare_sep(cpus=4)  # check multiprocessing
    get_meas_results(basic_measure, cpus=4)


def test_measure_kwargs():
    """Test measure kwargs parameters for sep"""
    target, results = get_meas_results(
        [sep_singleband_measure, sep_multiband_measure],
        measure_kwargs=[{"sigma_noise": 1.5}, {"sigma_noise": 2.0}],
    )
    detected_sources = {}
    for meas_function in [
        "sep_singleband_measure0",
        "sep_singleband_measure1",
        "sep_multiband_measure0",
        "sep_multiband_measure1",
    ]:
        assert meas_function in results["catalog"].keys()
        for i, blend in enumerate(results["catalog"][meas_function]):
            if len(blend) > 0:
                detected_centers = np.array([blend[0]["x_peak"].item(), blend[0]["y_peak"].item()])
                dist = np.max(np.abs(detected_centers - target[i]))
                np.testing.assert_array_less(dist, 1.5)
                if meas_function in detected_sources.keys():
                    detected_sources[meas_function] += 1
                else:
                    detected_sources[meas_function] = 1

    assert detected_sources["sep_multiband_measure0"] >= detected_sources["sep_singleband_measure0"]
    assert detected_sources["sep_multiband_measure1"] >= detected_sources["sep_singleband_measure1"]


def test_locations():
    # for 3 separate batches, test locations of drawn sources are correct.

    catalog_name = data_dir / "sample_input_catalog.fits"
    stamp_size = 24
    survey = get_surveys("LSST")
    catalog = CatsimCatalog.from_file(catalog_name)
    # make all catalog colulmns equally bright and detectable
    catalog.table["i_ab"] = 21
    catalog.table["r_ab"] = 21
    catalog.table["g_ab"] = 21
    catalog.table["z_ab"] = 21
    catalog.table["y_ab"] = 21
    catalog.table["u_ab"] = 21

    # with fixed centers
    for _ in range(3):
        shifts = [
            np.random.uniform(-2.5, 2.5, 2),
            np.random.uniform(-2.5, 2.5, 2),
            np.random.uniform(-2.5, 2.5, 2),
            np.random.uniform(-2.5, 2.5, 2),
        ]
        indexes = [[0], [1], [2], [3]]
        draw_blend_generator = CatsimGenerator(
            catalog,
            DefaultSampling(seed=TEST_SEED),
            [survey],
            shifts=shifts,
            indexes=indexes,
            stamp_size=stamp_size,
            seed=TEST_SEED,
            batch_size=4,
        )
        meas_generator = MeasureGenerator(
            sep_singleband_measure, draw_blend_generator, measure_kwargs=[{"sigma_noise": 1.5}]
        )

        blend_results, results = next(meas_generator)
        target_centers = np.array(
            [
                [blend["x_peak"].item(), blend["y_peak"].item()]
                for blend in blend_results["blend_list"]
            ]
        )
        # count the number of detected sources for both algroitms
        detected_sources = 0
        detected_cat = results["catalog"]["sep_singleband_measure"]
        for i, blend in enumerate(detected_cat):
            if len(blend) > 0:
                detected_centers = np.array([blend[0]["x_peak"].item(), blend[0]["y_peak"].item()])
                dist = np.max(np.abs(detected_centers - target_centers[i]))
                # make sure that detections are within 1.5 arcsec from truth
                np.testing.assert_array_less(dist, 1.5)

                # make sure returned true centers agree with input.
                np.testing.assert_array_almost_equal(
                    target_centers[i], shifts[i] / 0.2 + 60 - 0.5, decimal=6
                )
                detected_sources += 1

        assert detected_sources == len(target_centers)

    # with random centers, multiple batches
    for _ in range(3):
        draw_blend_generator = CatsimGenerator(
            catalog,
            DefaultSampling(seed=TEST_SEED, max_number=1),
            [survey],
            stamp_size=stamp_size,
            seed=TEST_SEED,
            batch_size=4,
        )
        meas_generator = MeasureGenerator(
            sep_singleband_measure, draw_blend_generator, measure_kwargs=[{"sigma_noise": 1.5}]
        )
        for _ in range(3):
            blend_results, results = next(meas_generator)
            target_centers = np.array(
                [
                    [blend["x_peak"].item(), blend["y_peak"].item()]
                    for blend in blend_results["blend_list"]
                ]
            )
            # count the number of detected sources for both algroitms
            detected_sources = 0
            detected_cat = results["catalog"]["sep_singleband_measure"]
            for i, blend in enumerate(detected_cat):
                if len(blend) > 0:
                    detected_centers = np.array(
                        [blend[0]["x_peak"].item(), blend[0]["y_peak"].item()]
                    )
                    dist = np.max(np.abs(detected_centers - target_centers[i]))
                    np.testing.assert_array_less(dist, 1.5)
                    detected_sources += 1
            assert detected_sources == len(target_centers)
