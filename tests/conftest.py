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
            [
                215.6290132,
                2394.11614647,
                8480.93459939,
                11069.50400858,
                8686.22664691,
                5538.14672976,
            ]
        )
        test_batch_mean = 7.354362014657712
        test_batch_std = 404.1066833449062
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
            [
                203.60154724,
                2072.32250977,
                7771.61767578,
                10532.68652344,
                8669.31933594,
                4871.0546875,
            ]
        )
        test_batch_mean = 3.7485726507963544
        test_batch_std = 92.11482419872036
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
