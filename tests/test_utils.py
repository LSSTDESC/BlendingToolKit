import os
import pytest
import astropy.table
import numpy as np
import btk.utils
import btk.config


def get_draw_generator():
    """Returns draw generator with group sampling function"""
    wld_catalog_name = 'data/sample_group_catalog.fits'
    catalog_name = 'data/sample_group_input_catalog.fits'
    param = btk.config.Simulation_params(catalog_name, max_number=10,
                                         batch_size=3)
    wld_catalog = astropy.table.Table.read(wld_catalog_name,
                                           format='fits')
    param.wld_catalog = wld_catalog
    param.group_id_count = 2
    np.random.seed(param.seed)
    catalog = btk.get_input_catalog.load_catalog(param)
    blend_generator = btk.create_blend_generator.generate(
        param, catalog, btk.utils.group_sampling_function_numbered)
    observing_generator = btk.create_observing_generator.generate(param)
    draw_blend_generator = btk.draw_blends.generate(
        param, blend_generator, observing_generator)
    return draw_blend_generator


@pytest.mark.timeout(5)
def test_group_sampling():
    """Test blends drawn with group sampling function with count"""
    draw_blend_generator = get_draw_generator()
    output = next(draw_blend_generator)
    blend_images = output['blend_images']
    batch_max = blend_images.max(axis=0).max(axis=0).max(axis=0)
    batch_mean = blend_images.mean()
    batch_std = blend_images.std()
    test_batch_max = np.array([378.6290132, 2082.11614647, 10042.93459939,
                               10939.50400858, 9472.22664691, 4909.14672976])
    test_batch_mean = 13.587452292435493
    test_batch_std = 719.954443492819
    np.testing.assert_array_almost_equal(
        batch_max, test_batch_max, decimal=3,
        err_msg="Did not get desired maximum pixel values of blend images")
    np.testing.assert_almost_equal(
        batch_mean, test_batch_mean, decimal=5,
        err_msg="Did not get desired mean pixel values of blend images")
    np.testing.assert_almost_equal(
        batch_std, test_batch_std, decimal=5,
        err_msg="Did not get desired std of pixel values of blend images")
