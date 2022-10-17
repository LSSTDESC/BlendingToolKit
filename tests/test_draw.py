import multiprocessing as mp
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import btk.plot_utils
from btk.survey import get_surveys

TEST_SEED = 0


def get_draw_generator(
    batch_size=8,
    cpus=1,
    add_noise="all",
    fixed_parameters=False,
    sampling_function=None,
):
    """Returns a btk.draw_blends generator for default parameters."""
    catalog_name = "data/sample_input_catalog.fits"
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
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    if sampling_function is None:
        sampling_function = btk.sampling_functions.DefaultSampling(
            stamp_size=stamp_size, seed=TEST_SEED
        )
    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        get_surveys("LSST"),
        batch_size=batch_size,
        stamp_size=stamp_size,
        shifts=shifts,
        indexes=indexes,
        cpus=cpus,
        add_noise=add_noise,
        verbose=True,
        seed=TEST_SEED,
    )
    return draw_generator


class TestMultiprocessing:
    def test_multiprocessing(self):
        b_size = 16
        cpus = np.min([mp.cpu_count(), 16])
        parallel_im_gen = get_draw_generator(b_size, cpus, add_noise="none")
        parallel_im = next(parallel_im_gen)
        serial_im_gen = get_draw_generator(b_size, cpus=1, add_noise="none")
        serial_im = next(serial_im_gen)
        np.testing.assert_array_equal(parallel_im["blend_images"], serial_im["blend_images"])
        np.testing.assert_array_equal(parallel_im["isolated_images"], serial_im["isolated_images"])
        next(parallel_im_gen)


class TestBasicDraw:
    @staticmethod
    def match_isolated_images_default(isolated_images):
        """Compares the max value of isolated image for each of the band along with
        the mean and std values in the batch. This is compared to the values
        measured a proiri for the default input settings.
        """
        test_batch_max = np.array([84.405, 1151.363, 7264.221, 9658.402, 7878.188, 4316.912])
        test_batch_mean = 3.1129989295360363
        test_batch_std = 87.75473356357536
        batch_max = isolated_images.max(axis=(0, 1, 3, 4))
        batch_mean = isolated_images.mean()
        batch_std = isolated_images.std()
        np.testing.assert_array_almost_equal(
            batch_max,
            test_batch_max,
            decimal=3,
            err_msg="Did not get desired maximum pixel values of isolated images",
        )
        np.testing.assert_almost_equal(
            batch_mean,
            test_batch_mean,
            decimal=3,
            err_msg="Did not get desired mean pixel values of isolated images",
        )
        np.testing.assert_almost_equal(
            batch_std,
            test_batch_std,
            decimal=3,
            err_msg="Did not get desired std of pixel values of isolated images",
        )

    @staticmethod
    def match_blend_images_default(blend_images):
        """Compares the max value of blend image for each of the band along with
        the mean and std values in the batch. This is compared to the values
        measured a priori for the default input settings.
        """
        test_batch_max = np.array([217.32, 1316.92, 7756.88, 10434.56, 8382.2, 4928.28])
        test_batch_mean = 6.094021990717676
        test_batch_std = 402.6837009646611
        batch_max = blend_images.max(axis=(0, 2, 3))
        batch_mean = blend_images.mean()
        batch_std = blend_images.std()
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

    @staticmethod
    def match_background_noise(blend_images):
        """Computes the background noise value of second blend scene image for in
        the r band. This is compared to the values measured a priori for the
        default input settings.
        """
        test_batch_noise = 128666.38136196136
        batch_noise = np.var(blend_images[1, 2, 0:32, 0:32])
        np.testing.assert_almost_equal(
            batch_noise,
            test_batch_noise,
            decimal=3,
            err_msg="Did not get desired mean pixel values of blend images",
        )

    def test_basic_sampling(self):
        sampling_function = btk.sampling_functions.BasicSampling()
        draw_generator = get_draw_generator(
            fixed_parameters=True, sampling_function=sampling_function
        )
        next(draw_generator)

    @patch("btk.plot_utils.plt.show")
    def test_default(self, mock_show):
        default_draw_generator = get_draw_generator(fixed_parameters=True)
        draw_output = next(default_draw_generator)
        btk.plot_utils.plot_blends(draw_output["blend_images"], draw_output["blend_list"])
        plt.close("all")
        btk.plot_utils.plot_with_isolated(
            draw_output["blend_images"], draw_output["isolated_images"], draw_output["blend_list"]
        )
        plt.close("all")
        assert len(draw_output["blend_list"]) == 8, "Default batch should return 8"
        assert (
            len(draw_output["blend_list"][3]) < 3
        ), "Default max_number should \
            generate 2 or 1 galaxies per blend."
        self.match_background_noise(draw_output["blend_images"])
        self.match_isolated_images_default(draw_output["isolated_images"])
        self.match_blend_images_default(draw_output["blend_images"])


def test_shear_draw():
    stamp_size = 24.0
    seed = 0
    catalog_name = "data/sample_input_catalog.fits"
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    sampling_function_shear = btk.sampling_functions.DefaultSamplingShear(
        stamp_size=stamp_size, shear=(0.5, 0), seed=seed, max_number=3, min_number=3
    )
    survey = btk.survey.get_surveys("LSST")
    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function_shear,
        survey,
        batch_size=10,
        stamp_size=stamp_size,
        cpus=1,
        add_noise="all",
        seed=seed,
        apply_shear=True,
    )
    next(draw_generator)


def test_rotation():
    stamp_size = 24.0
    seed = 0
    catalog_name = "data/sample_input_catalog.fits"
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    sampling_function_shear = btk.sampling_functions.DefaultSampling(
        stamp_size=stamp_size, seed=seed, max_number=3, min_number=1
    )
    survey = btk.survey.get_surveys("LSST")
    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function_shear,
        survey,
        batch_size=10,
        stamp_size=stamp_size,
        cpus=1,
        add_noise="all",
        seed=seed,
        augment_data=True,
    )
    blend_results = next(draw_generator)
    rotation = blend_results["blend_list"][0]["btk_rotation"]
    assert np.all(0 <= rotation) and np.all(rotation <= 360)
