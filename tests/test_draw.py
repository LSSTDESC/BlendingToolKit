import numpy as np
import pytest
import btk
import btk.config
import multiprocessing as mp


def get_draw_generator(batch_size=8, cpus=1,
                       multiprocessing=False, add_noise=True):
    """Returns a btk.draw_blends generator for default parameters"""
    catalog_name = 'data/sample_input_catalog.fits'
    param = btk.config.Simulation_params(catalog_name, batch_size=batch_size,
                                         add_noise=add_noise)
    np.random.seed(param.seed)
    catalog = btk.get_input_catalog.load_catalog(param)
    blend_generator = btk.create_blend_generator.generate(param, catalog)
    observing_generator = btk.create_observing_generator.generate(param)
    draw_generator = btk.draw_blends.generate(param, blend_generator,
                                              observing_generator,
                                              multiprocessing=multiprocessing,
                                              cpus=cpus)
    return draw_generator


def match_blend_images_default(blend_images):
    """Compares the max value of blend image for each of the band along with
    the mean and std values in the batch. This is compared to the values
    measured a priori for the default input settings.
    """
    test_batch_max = np.array([215.6290132, 2394.11614647, 8480.93459939,
                               11069.50400858, 8686.22664691, 5538.14672976])
    test_batch_mean = 7.354362014657712
    test_batch_std = 404.1066833449062
    batch_max = blend_images.max(axis=0).max(axis=0).max(axis=0)
    batch_mean = blend_images.mean()
    batch_std = blend_images.std()
    np.testing.assert_array_almost_equal(
        batch_max, test_batch_max, decimal=3,
        err_msg="Did not get desired maximum pixel values of blend images")
    np.testing.assert_almost_equal(
        batch_mean, test_batch_mean, decimal=5,
        err_msg="Did not get desired mean pixel values of blend images")
    np.testing.assert_almost_equal(
        batch_std, test_batch_std, decimal=5,
        err_msg="Did not get desired std of pixel values of blend images")


def match_isolated_images_default(isolated_images):
    """Compares the max value of isolated image for each of the band along with
    the mean and std values in the batch. This is compared to the values
    measured a proiri for the default input settings.
    """
    test_batch_max = np.array([203.60154724, 2072.32250977, 7771.61767578,
                               10532.68652344, 8669.31933594, 4871.0546875])
    test_batch_mean = 3.7485726507963544
    test_batch_std = 92.11482419872036
    batch_max = isolated_images.max(axis=0).max(axis=0).max(axis=0).max(axis=0)
    batch_mean = isolated_images.mean()
    batch_std = isolated_images.std()
    np.testing.assert_array_almost_equal(
        batch_max, test_batch_max, decimal=3,
        err_msg="Did not get desired maximum pixel values of isolated images")
    np.testing.assert_almost_equal(
        batch_mean, test_batch_mean, decimal=5,
        err_msg="Did not get desired mean pixel values of isolated images")
    np.testing.assert_almost_equal(
        batch_std, test_batch_std, decimal=5,
        err_msg="Did not get desired std of pixel values of isolated images")


def match_background_noise(blend_images):
    """Computes the background noise value of second blend scene image for in
    the i band. This is compared to the values measured a priori for the
    default input settings.
    """
    test_batch_noise = 176958.46899032593
    batch_noise = np.var(blend_images[1, 0:32, 0:32, 3])
    np.testing.assert_almost_equal(
        batch_noise, test_batch_noise, decimal=5,
        err_msg="Did not get desired mean pixel values of blend images")


@pytest.mark.timeout(5)
def test_default():
    default_draw_generator = get_draw_generator()
    draw_output = next(default_draw_generator)
    assert len(draw_output['blend_list']) == 8, "Default batch should return 8"
    assert len(draw_output['blend_list'][3]) < 3, "Default max_number should \
        generate 2 or 1 galaxies per blend."
    assert draw_output['obs_condition'][5][0].survey_name == 'LSST', "Default \
        observing survey is LSST."
    match_blend_images_default(draw_output['blend_images'])
    match_isolated_images_default(draw_output['isolated_images'])
    match_background_noise(draw_output['blend_images'])
    pass


@pytest.mark.timeout(15)
def test_multi_processing():
    b_size = 16
    try:
        cpus = np.min([mp.cpu_count(), 16])
    except NotImplementedError:
        cpus = 2
    parallel_im_gen = get_draw_generator(b_size, cpus, multiprocessing=True,
                                         add_noise=False)
    parallel_im = next(parallel_im_gen)
    serial_im_gen = get_draw_generator(b_size, cpus, multiprocessing=False,
                                       add_noise=False)
    serial_im = next(serial_im_gen)
    np.testing.assert_array_equal(parallel_im['blend_images'],
                                  serial_im['blend_images'])
    np.testing.assert_array_equal(parallel_im['isolated_images'],
                                  serial_im['isolated_images'])
    pass
