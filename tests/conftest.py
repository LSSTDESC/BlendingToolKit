import pytest
import numpy as np


@pytest.fixture(scope="session")
def match_images():
    return Match_Images


class Match_Images(object):
    @staticmethod
    def match_blend_images_default(blend_images):
        """Compares the max value of blend image for each of the band along with
        the mean and std values in the batch. This is compared to the values
        measured a priori for the default input settings.
        """
        test_batch_max = np.array(
            [5428.147, 8947.227, 11190.504, 8011.935, 1536.116, 191.629]
        )
        test_batch_mean = 5.912076135028083
        test_batch_std = 403.5577217178115
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
            [4772.817, 8506.056, 10329.56, 7636.189, 1245.693, 90.721]
        )
        test_batch_mean = 3.1101762559117585
        test_batch_std = 90.74182140645624
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
