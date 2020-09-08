import pytest
import numpy as np


@pytest.fixture(scope="session")
def input_args():
    return Input_Args


@pytest.fixture(scope="session")
def match_images():
    return Match_Images


class Input_Args(object):
    """Class that returns values in the same format as argparse in btk_input."""

    def __init__(
        self,
        simulation="two_gal",
        name="unit_test",
        configfile="tests/test-config.yaml",
        verbose=True,
    ):
        self.simulation = simulation
        self.configfile = configfile
        self.name = name
        self.verbose = verbose


class Match_Images(object):
    @staticmethod
    def match_blend_images_default(blend_images):
        """Compares the max value of blend image for each of the band along with
        the mean and std values in the batch. This is compared to the values
        measured a priori for the default input settings.
        """
        test_batch_max = np.array(
            [5352.147, 8842.227, 11197.504, 8216.935, 1535.116, 191.629]
        )
        test_batch_mean = 5.6334577901206755
        test_batch_std = 403.5918774641063
        batch_max = blend_images.max(axis=0).max(axis=0).max(axis=0)
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
            decimal=5,
            err_msg="Did not get desired mean pixel values of blend images",
        )
        np.testing.assert_almost_equal(
            batch_std,
            test_batch_std,
            decimal=5,
            err_msg="Did not get desired std of pixel values of blend images",
        )

    @staticmethod
    def match_isolated_images_default(isolated_images):
        """Compares the max value of isolated image for each of the band along with
        the mean and std values in the batch. This is compared to the values
        measured a proiri for the default input settings.
        """
        test_batch_max = np.array(
            [4774.28, 8503.917, 10339.48, 7637.224, 1244.871, 90.719]
        )
        test_batch_mean = 2.9643491324620252
        test_batch_std = 90.7578274135593
        batch_max = isolated_images.max(axis=0).max(axis=0).max(axis=0).max(axis=0)
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
            decimal=5,
            err_msg="Did not get desired mean pixel values of isolated images",
        )
        np.testing.assert_almost_equal(
            batch_std,
            test_batch_std,
            decimal=5,
            err_msg="Did not get desired std of pixel values of isolated images",
        )
