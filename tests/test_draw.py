import multiprocessing as mp
import numpy as np
import pytest
import btk
import btk.sampling_functions
import btk.obs_conditions
from btk.obs_conditions import Rubin, HSC


def get_draw_generator(
    batch_size=8, cpus=1, multiprocessing=False, add_noise=True, fixed_parameters=False
):
    """Returns a btk.draw_blends generator for default parameters"""
    catalog_name = "data/sample_input_catalog.fits"

    np.random.seed(0)
    stamp_size = 24.0
    if fixed_parameters:
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
    else:
        shifts = None
        indexes = None
    catalog = btk.catalog.WLDCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)
    survey = btk.obs_conditions.Rubin
    obs_conds = btk.obs_conditions.WLDObsConditions(stamp_size)
    draw_generator = btk.draw_blends.WLDGenerator(
        catalog,
        sampling_function,
        survey,
        obs_conds=obs_conds,
        batch_size=batch_size,
        stamp_size=stamp_size,
        shifts=shifts,
        indexes=indexes,
        multiprocessing=multiprocessing,
        cpus=cpus,
        add_noise=add_noise,
    )
    return draw_generator


def match_background_noise(blend_images):
    """Computes the background noise value of second blend scene image for in
    the i band. This is compared to the values measured a priori for the
    default input settings.
    """
    test_batch_noise = 129661.1961517334
    batch_noise = np.var(blend_images[1, 0:32, 0:32, 3])
    np.testing.assert_almost_equal(
        batch_noise,
        test_batch_noise,
        decimal=5,
        err_msg="Did not get desired mean pixel values of blend images",
    )


@pytest.mark.timeout(10)
def test_default(match_images):
    default_draw_generator = get_draw_generator(fixed_parameters=True)
    draw_output = next(default_draw_generator)
    assert len(draw_output["blend_list"]) == 8, "Default batch should return 8"
    assert (
        len(draw_output["blend_list"][3]) < 3
    ), "Default max_number should \
        generate 2 or 1 galaxies per blend."
    assert (
        draw_output["obs_condition"][0].survey_name == "LSST"
    ), "Default observing survey is LSST."
    match_images.match_blend_images_default(draw_output["blend_images"])
    match_images.match_isolated_images_default(draw_output["isolated_images"])
    match_background_noise(draw_output["blend_images"])


@pytest.mark.timeout(15)
def test_multi_processing():
    b_size = 16
    try:
        cpus = np.min([mp.cpu_count(), 16])
    except NotImplementedError:
        cpus = 2
    parallel_im_gen = get_draw_generator(
        b_size, cpus, multiprocessing=True, add_noise=False
    )
    parallel_im = next(parallel_im_gen)
    serial_im_gen = get_draw_generator(
        b_size, cpus, multiprocessing=False, add_noise=False
    )
    serial_im = next(serial_im_gen)
    np.testing.assert_array_equal(
        parallel_im["blend_images"], serial_im["blend_images"]
    )
    np.testing.assert_array_equal(
        parallel_im["isolated_images"], serial_im["isolated_images"]
    )
    pass


@pytest.mark.timeout(10)
def test_multiresolution():
    catalog_name = "data/sample_input_catalog.fits"

    np.random.seed(0)
    stamp_size = 24.0
    batch_size = 8
    cpus = 1
    multiprocessing = False
    add_noise = True

    catalog = btk.catalog.WLDCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)
    obs_conds = btk.obs_conditions.WLDObsConditions(stamp_size)
    draw_generator = btk.draw_blends.WLDGenerator(
        catalog,
        sampling_function,
        [Rubin, HSC],
        obs_conds=obs_conds,
        stamp_size=stamp_size,
        batch_size=batch_size,
        multiprocessing=multiprocessing,
        cpus=cpus,
        add_noise=add_noise,
        meas_bands=("i", "i"),
    )
    draw_output = next(draw_generator)

    assert (
        "LSST" in draw_output["blend_list"].keys()
    ), "Both surveys get well defined outputs"
    assert (
        "HSC" in draw_output["blend_list"].keys()
    ), "Both surveys get well defined outputs"
    assert draw_output["blend_images"]["LSST"][0].shape[0] == int(
        24.0 / 0.2
    ), "LSST survey should have a pixel scale of 0.2"
    assert draw_output["blend_images"]["HSC"][0].shape[0] == int(
        24.0 / 0.167
    ), "HSC survey should have a pixel scale of 0.167"
