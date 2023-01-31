"""Atomic metric functions."""
import numpy as np
import skimage


def mse(image1: np.ndarray, image2: np.ndarray):
    """Computes mean-squared-error (MSE) between two images.

    Args:
        images1: Array of shape `NCHW` containing `N` images each of
                        size `CHW`.
        images2: Array of shape `NCHW` containing `N` images each of
                        size `CHW`.

    Returns:
        Returns MSE between each corresponding iamge as an array of shape `NC`.
    """
    return np.power((image1 - image2), 2).mean(axis=(-1, -2)).sqrt()


def iou(
    segmentation1: np.ndarray,
    segmentation2: np.ndarray,
):
    """Calculates intersection-over-union (IoU) given two semgentation arrays.

    The segmentation arrays should each have values of 1 or 0s only.

    Args:
        segmentation1: Array of shape NHW containing `N` segmentation maps each of
                        size `HW`.
        segmentation2: Array of shape NHW containing `N` segmentation maps each of
                        size `HW`.

    Returns:
        Returns `iou` between each corresponding segmentation map as an array of shape `N`

    """
    seg1 = segmentation1.astype(bool)
    seg2 = segmentation2.astype(bool)
    i = np.logical_and(seg1, seg2).sum(axis=(-1, -2))
    u = np.logical_or(seg1, seg2).sum(axis=(-1, -2))
    return i / u


def psnr(image1: np.ndarray, image2: np.ndarray):
    """Compute peak-signal-to-noise-ratio."""
    # TODO: use data_range?
    # compute in batches or over number of sources?
    return skimage.metrics.peak_signal_noise_ratio(image1, image2)


def ssim(image1: np.ndarray, image2: np.ndarray):
    """Compute structural similarity index."""
    return skimage.metrics.structural_similarity(
        np.moveaxis(image1, 0, -1),
        np.moveaxis(image2, 0, -1),
        channel_axis=-1,
    )
