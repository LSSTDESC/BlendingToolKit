import os
import numpy as np
#import sys
import pytest
#import astropy.table
#parentdir = os.path.dirname(os.getcwd())
#sys.path.insert(0, parentdir)
import btk
import btk.config


def get_draw_generator(batch_size=8, cpus=1, multiprocessing=False):
    catalog_name = 'sample_input_catalog.fits'
    param = btk.config.Simulation_params(catalog_name, batch_size=batch_size)
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
    measured a proiri for the default input settings.
    """
    test_batch_max = np.array([2525.62890625,  55283.1171875, 281863.9375,
                               324827.5, 250068.234375, 141810.15625])
    test_batch_mean = 266.2592160488058
    test_batch_std = 4185.675821393696
    batch_max = blend_images.max(axis=0).max(axis=0).max(axis=0)
    batch_mean = blend_images.mean()
    batch_std = blend_images.std()
    np.testing.assert_array_almost_equal(
        batch_max, test_batch_max,
        err_msg="Did not get desired maximum pixel values of blend images")
    np.testing.assert_almost_equal(
        test_batch_mean, batch_mean,
        err_msg="Did not get desired mean pixel valuesof blend images")
    np.testing.assert_almost_equal(
        test_batch_std, batch_std,
        err_msg="Did not get desired std of pixel valuesof blend images")


def match_isolated_images_default(isolated_images):
    """Compares the max value of isoalted image for each of the band along with
    the mean and std values in the batch. This is compared to the values
    measured a proiri for the default input settings.
    """
    test_batch_max = np.array([2469.87939453, 55053.98828125, 282896.9375,
                               324907.90625, 250092.578125, 141441.015625])
    test_batch_mean = 133.4565486863647
    test_batch_std = 2950.5856685453587
    batch_max = isolated_images.max(axis=0).max(axis=0).max(axis=0)
    batch_mean = isolated_images.mean()
    batch_std = isolated_images.std()
    np.testing.assert_array_almost_equal(
        batch_max, test_batch_max,
        err_msg="Did not get desired maximum pixel values of isolated images")
    np.testing.assert_almost_equal(
        test_batch_mean, batch_mean,
        err_msg="Did not get desired mean pixel values of isolated images")
    np.testing.assert_almost_equal(
        test_batch_std, batch_std,
        err_msg="Did not get desired std of pixel values of isolated images")


@pytest.mark.timeout(5)
def test_default():
    default_draw_generator = get_draw_generator()
    draw_output = next(default_draw_generator)
    match_blend_images_default(draw_output['blend_images'])
    match_isolated_images_default(draw_output['isolated_images'])
    assert len(draw_output['blend_list']) == 8, "Default batch should return 8"
    assert len(draw_output['blend_list'][3]) < 3, "Default max_number should \
        generate 2 or 1 galaxies per blend."
    assert draw_output['obs_condition'][5][0].survey_name == 'LSST', "Default \
        observing survey is LSST."

    pass


@pytest.mark.timeout(15)
def test_multi_processing():
    b_size = 16
    cpus = os.cpu_count()
    parallel_im_gen = get_draw_generator(b_size, cpus, multiprocessing=True)
    parallel_im = next(parallel_im_gen)
    serial_im_gen = get_draw_generator(b_size, cpus, multiprocessing=False)
    serial_im = next(serial_im_gen)
    print(parallel_im['blend_images'].shape, serial_im['blend_images'].shape)
    np.testing.assert_array_equal(parallel_im['blend_images'],
                                  serial_im['blend_images'])
    np.testing.assert_array_equal(parallel_im['isolated_images'],
                                  serial_im['isolated_images'])
    pass
