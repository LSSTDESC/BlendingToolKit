"""Atomic metric functions."""


import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_segmentation(isolated_images: np.ndarray, sky_level: float, sigma_noise: float = 3):
    """Get segmentation from isolated galaxy images based on noise level cutoff.

    Input of `isolated_images` should be a single band.

    Args:
        isolated_images: Images of isolated galaxies. Shape only be in a single band.
        sky_level: Background level of all images. Images are assume to not be
            background-substracted.
        sigma_noise: Sigma level at which an pixel is considered to be noise. Should match
            the argument in SEP if that is being used to compuate the segmentation
            for deblending.
    """
    assert isolated_images.ndim == 4
    _, _, h, w = isolated_images.shape
    err = np.sqrt(sky_level)
    is_on = isolated_images.sum(axis=(-1, -2)) > 0
    is_on = is_on[:, :, None, None].repeat(h, axis=2).repeat(w, axis=3)
    seg = isolated_images > sigma_noise * err
    assert is_on.shape == seg.shape
    return np.where(is_on, seg, np.nan)


def mse(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Computes mean-squared-error (MSE) between two images.

    Args:
        images1: Array of shape `*HW` containing `*` images each of
                        size `*HW`.
        images2: Array of shape `*HW` containing `*` images each of
                        size `*HW`.

    Returns:
        Returns MSE between each corresponding iamge as an array of shape `*`.
    """
    return np.sqrt(np.power((image1 - image2), 2).mean(axis=(-1, -2)))


def iou(seg1: np.ndarray, seg2: np.ndarray) -> np.ndarray:
    """Calculates intersection-over-union (IoU) given two semgentation arrays.

    The segmentation arrays should each have values of 1 or 0s only.

    Args:
        seg1: Array of shape `NHW` containing `N` segmentation maps each of
                        size `HW`.
        seg2: Array of shape `NHW` containing `N` segmentation maps each of
                        size `HW`.

    Returns:
        Returns `iou` between each corresponding segmentation map as an array of shape `N`

    """
    assert not np.any(np.isnan(seg1)) and not np.any(np.isnan(seg2))
    seg1 = seg1.astype(bool)
    seg2 = seg1.astype(bool)
    i = np.logical_and(seg1, seg2).sum(axis=(-1, -2))
    u = np.logical_or(seg1, seg2).sum(axis=(-1, -2))
    return i / u


def psnr(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Compute peak-signal-to-noise-ratio from skimage.

    Args:
        images1: Array of shape `NHW` containing `N` images each of
                size `NHW`.
        images2: Array of shape `NHW` containing `N` images each of
                size `NHW`.

    Returns:
        Returns PSNR between each corresponding iamge as an array of shape `N`.
    """
    n, h, w = image1.shape
    assert image1.min() >= 0 and image2.min() >= 0
    assert (n, h, w) == image2.shape
    psnr_ = np.zeros(n)
    for ii in range(n):
        im1 = image1[ii] / image1[ii].max()
        im2 = image2[ii] / image2[ii].max()
        psnr_[ii] = peak_signal_noise_ratio(im1, im2, data_range=1)
    return psnr_


def struct_sim(image1: np.ndarray, image2: np.ndarray, **kwargs) -> np.ndarray:
    """Compute structural similarity index from skimage.

    By default, we normalize the images to be between 0 and 1. So that the
    `data_range` is 1.

    Args:
        images1: Array of shape `NHW` containing `N` images each of
                        size `NHW`.
        images2: Array of shape `NHW` containing `N` images each of
                        size `NHW`.
        kwargs: Keyword arguments to be passed in to `peak_signal_noise_ratio` function
                within `skimage.metrics`.

    Returns:
        Returns structural similarity index between each corresponding iamge as
        an array of shape `N`.
    """
    assert image1.min() >= 0 and image2.min() >= 0
    n, h, w = image1.shape
    assert (n, h, w) == image2.shape
    ssim = np.zeros(n)
    for ii in range(n):
        im1 = image1[ii] / image1[ii].max()
        im2 = image2[ii] / image2[ii].max()
        ssim[ii] = structural_similarity(im1, im2, data_range=1, **kwargs)
    return ssim


def effmat(tp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Returns efficiency matrices based on number of matched truth and predicted galaxies."""
    n = len(t)  # batch size
    max_true_n_sources = max(t)
    max_pred_n_sources = max(tp)
    eff_mat = np.zeros((max_pred_n_sources + 1, max_true_n_sources + 1))
    for ii in range(n):
        eff_mat[tp[ii], t[ii]] += 1
    return eff_mat
