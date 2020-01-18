import pytest
import astropy.table
import numpy as np
import btk.utils
import btk.config


def get_draw_generator(batch_size=3):
    """Returns draw generator with group sampling function"""
    wld_catalog_name = 'data/sample_group_catalog.fits'
    catalog_name = 'data/sample_group_input_catalog.fits'
    param = btk.config.Simulation_params(catalog_name, max_number=10,
                                         batch_size=batch_size)
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


def get_meas_generator(meas_params):
    """Returns draw generator with group sampling function"""
    catalog_name = 'data/sample_input_catalog.fits'
    param = btk.config.Simulation_params(catalog_name, batch_size=1,
                                         add_noise=True)
    np.random.seed(param.seed)
    catalog = btk.get_input_catalog.load_catalog(param)
    blend_generator = btk.create_blend_generator.generate(param, catalog)
    observing_generator = btk.create_observing_generator.generate(param)
    draw_generator = btk.draw_blends.generate(param, blend_generator,
                                              observing_generator)
    meas_generator = btk.measure.generate(
        meas_params, draw_generator, param)
    return meas_generator, param


@pytest.mark.timeout(5)
def test_group_sampling():
    """Test blends drawn with group sampling function"""
    draw_blend_generator = get_draw_generator()
    output = next(draw_blend_generator)
    blend_images = output['blend_images']
    batch_max = blend_images.max(axis=0).max(axis=0).max(axis=0)
    batch_mean = blend_images.mean()
    batch_std = blend_images.std()
    test_batch_max = np.array([378.6290132, 2082.11614647, 10042.93459939,
                               10939.50400858, 9472.22664691, 4909.14672976])
    test_batch_mean = 13.589091952929321
    test_batch_std = 719.7592990809109
    np.testing.assert_array_almost_equal(
        batch_max, test_batch_max, decimal=3,
        err_msg="Did not get desired maximum pixel values of blend images")
    np.testing.assert_almost_equal(
        batch_mean, test_batch_mean, decimal=5,
        err_msg="Did not get desired mean pixel values of blend images")
    np.testing.assert_almost_equal(
        batch_std, test_batch_std, decimal=5,
        err_msg="Did not get desired std of pixel values of blend images")


def compare_sep():
    """Test detection with sep"""
    meas_param = btk.utils.SEP_params()
    meas_generator, param = get_meas_generator(meas_param)
    output, deb, _ = next(meas_generator)
    detected_centers = deb[0]['peaks']
    target_detection = np.array([[64.62860131, 61.83551097]])
    np.testing.assert_array_almost_equal(
        detected_centers, target_detection, decimal=3,
        err_msg="Did not get desired detections")
    pass


def compare_stack():
    """Test detection with stack"""
    pass


def compare_scarlet():
    """Test deblending with scarlet"""
    meas_param = btk.utils.Scarlet_params()
    meas_generator, param = get_meas_generator(meas_param)
    output, deb, _ = next(meas_generator)
    blend_list = output['blend_list']
    deblend_images = [deb[i]['deblend_image'] for i in range(len(blend_list))]
    batch_max = deblend_images[0].max(axis=0).max(axis=0).max(axis=0)
    batch_mean = deblend_images[0].mean()
    batch_std = deblend_images[0].std()
    test_batch_max = np.array([10.39827598, 279.24613539, 1511.07999549,
                               1083.94685111, 567.58024363, 403.28130687])
    test_batch_mean = 4.09093024221
    test_batch_std = 41.334411867967
    np.testing.assert_array_almost_equal(
        batch_max, test_batch_max, decimal=3,
        err_msg="Did not get desired maximum pixel values of deblend images")
    np.testing.assert_almost_equal(
        batch_mean, test_batch_mean, decimal=5,
        err_msg="Did not get desired mean pixel values of deblend images")
    np.testing.assert_almost_equal(
        batch_std, test_batch_std, decimal=5,
        err_msg="Did not get desired std of pixel values of deblend images")
    pass


@pytest.mark.timeout(15)
def test_algorithms():
    """Test detection/deblending/measurement algorithms if installed"""
    try:
        import sep
        compare_sep()
    except ModuleNotFoundError:
        print("skipping sep test")
    try:
        import scarlet
        #compare_scarlet()
    except ModuleNotFoundError:
        print("skipping scarlet test")
    try:
        import lsst.afw.table
        compare_stack()
    except ModuleNotFoundError:
        print("skipping stack test")
