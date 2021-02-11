import numpy as np
import btk.sampling_functions
import btk.survey


def get_draw_generator(batch_size=3):
    """Returns draw generator with group sampling function"""
    wld_catalog_name = "data/sample_group_catalog.fits"
    catalog_name = "data/sample_group_input_catalog.fits"

    max_number = 10
    stamp_size = 24
    survey = btk.survey.Rubin
    pixel_scale = 0.2
    shift = [0.8, -0.7]
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.GroupSamplingFunctionNumbered(
        max_number, wld_catalog_name, stamp_size, pixel_scale, shift=shift
    )
    draw_blend_generator = btk.draw_blends.CatsimGenerator(
        catalog, sampling_function, [survey], batch_size=batch_size
    )
    return draw_blend_generator


def get_meas_generator(meas_params, multiprocessing=False, cpus=1):
    """Returns draw generator with group sampling function"""

    catalog_name = "data/sample_input_catalog.fits"
    np.random.seed(0)
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
        meas_params, draw_blend_generator, multiprocessing=multiprocessing, cpus=cpus
    )
    return meas_generator


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
    meas_param = btk.measure.SEP_params()
    meas_generator = get_meas_generator(meas_param)
    _, deb, _ = next(meas_generator)
    detected_centers = deb[0]["peaks"]
    target_detection = np.array([[65.588, 50.982]])
    np.testing.assert_array_almost_equal(
        detected_centers,
        target_detection,
        decimal=3,
        err_msg="Did not get desired detections",
    )


def compare_sep_multiprocessing():
    """Test detection with sep"""
    meas_param = btk.measure.SEP_params()
    meas_generator = get_meas_generator(meas_param, multiprocessing=True, cpus=4)
    _, deb, _ = next(meas_generator)
    detected_centers = deb[0]["peaks"]
    target_detection = np.array([[65.588, 50.982]])
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
