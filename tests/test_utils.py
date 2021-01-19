import pytest
import numpy as np
import btk.utils
import btk.sampling_functions
import btk.obs_conditions


def get_draw_generator(batch_size=3):
    """Returns draw generator with group sampling function"""
    wld_catalog_name = "data/sample_group_catalog.fits"
    catalog_name = "data/sample_group_input_catalog.fits"

    max_number = 10
    stamp_size = 24
    survey = btk.obs_conditions.Rubin
    pixel_scale = 0.2
    shift = [0.8, -0.7]
    np.random.seed(0)
    catalog = btk.catalog.WLDCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.GroupSamplingFunctionNumbered(
            max_number, wld_catalog_name, stamp_size, pixel_scale, shift=shift
        )
    obs_conds = btk.obs_conditions.WLDObsConditions(stamp_size)
    draw_blend_generator = btk.draw_blends.WLDGenerator(
        catalog,sampling_function,survey,obs_conds=obs_conds
    )
    return draw_blend_generator


def get_meas_generator(meas_params, multiprocessing=False, cpus=1):
    """Returns draw generator with group sampling function"""

    catalog_name = "data/sample_input_catalog.fits"
    np.random.seed(0)
    stamp_size = 24
    survey = btk.obs_conditions.Rubin
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
    catalog = btk.catalog.WLDCatalog.from_file(catalog_name)
    obs_conds = btk.obs_conditions.WLDObsConditions(stamp_size)
    draw_blend_generator = btk.draw_blends.WLDGenerator(
        catalog,
        btk.sampling_functions.DefaultSampling(),
        survey,
        obs_conds=obs_conds,
        shifts=shifts,
        indexes=indexes
    )
    meas_generator = btk.measure.MeasureGenerator(
        meas_params, draw_blend_generator, multiprocessing=multiprocessing, cpus=cpus
    )
    return meas_generator


@pytest.mark.timeout(15)
def test_group_sampling():
    """Test blends drawn with group sampling function"""
    draw_blend_generator = get_draw_generator()
    output = next(draw_blend_generator)
    blend_images = output["blend_images"]
    batch_max = blend_images.max(axis=0).max(axis=0).max(axis=0)
    batch_mean = blend_images.mean()
    batch_std = blend_images.std()
    test_batch_max = np.array(
        [17095.147, 30909.227, 44017.504, 44033.935, 14230.116, 1198.629]
    )
    test_batch_mean = 82.10116371218854
    test_batch_std = 1027.5460941593055
    np.testing.assert_array_almost_equal(
        batch_max,
        test_batch_max,
        decimal=3,
        err_msg="Did not get desired maximum pixel values of blend images",
    )
    np.testing.assert_almost_equal(
        batch_mean,
        test_batch_mean,
        decimal=3,
        err_msg="Did not get desired mean pixel values of blend images",
    )
    np.testing.assert_almost_equal(
        batch_std,
        test_batch_std,
        decimal=3,
        err_msg="Did not get desired std of pixel values of blend images",
    )


def compare_sep():
    """Test detection with sep"""
    meas_param = btk.utils.SEP_params()
    meas_generator = get_meas_generator(meas_param)
    output, deb, _ = next(meas_generator)
    detected_centers = deb[0]["peaks"]
    target_detection = np.array([[65.588, 50.982]])
    np.testing.assert_array_almost_equal(
        detected_centers,
        target_detection,
        decimal=3,
        err_msg="Did not get desired detections",
    )
    pass


def compare_sep_multiprocessing():
    """Test detection with sep"""
    meas_param = btk.utils.SEP_params()
    meas_generator = get_meas_generator(meas_param, multiprocessing=True, cpus=4)
    output, deb, _ = next(meas_generator)
    detected_centers = deb[0]["peaks"]
    target_detection = np.array([[65.588, 50.982]])
    np.testing.assert_array_almost_equal(
        detected_centers,
        target_detection,
        decimal=3,
        err_msg="Did not get desired detections",
    )
    pass


def compare_stack():
    """Test detection with stack"""
    pass


def compare_scarlet():
    """Test deblending with scarlet"""
    meas_param = btk.utils.Scarlet_params()
    meas_generator = get_meas_generator(meas_param)
    output, deb, _ = next(meas_generator)
    blend_list = output["blend_list"]
    deblend_images = [deb[i]["deblend_image"] for i in range(len(blend_list))]
    deblend_images[0].max(axis=0).max(axis=0).max(axis=0)
    deblend_images[0].mean()
    deblend_images[0].std()


def compare_scarlet_multiprocessing():
    """Test deblending with scarlet"""
    meas_param = btk.utils.Scarlet_params()
    meas_generator = get_meas_generator(meas_param, multiprocessing=True, cpus=4)
    output, deb, _ = next(meas_generator)
    blend_list = output["blend_list"]
    deblend_images = [deb[i]["deblend_image"] for i in range(len(blend_list))]
    deblend_images[0].max(axis=0).max(axis=0).max(axis=0)
    deblend_images[0].mean()
    deblend_images[0].std()


@pytest.mark.timeout(35)
def test_algorithms():
    """Test detection/deblending/measurement algorithms if installed"""
    try:
        import sep

        compare_sep()
        compare_sep_multiprocessing()
    except ModuleNotFoundError:
        print("skipping sep test")
    try:
        import scarlet

        compare_scarlet()
        compare_scarlet_multiprocessing()
    except ModuleNotFoundError:
        print("skipping scarlet test")
    try:
        import lsst.afw.table

        compare_stack()
    except ModuleNotFoundError:
        print("skipping stack test")
