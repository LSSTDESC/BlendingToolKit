"""Atomic metric functions."""

from typing import List

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_segmentation(isolated_images: np.ndarray, sky_level: float, sigma_noise: float = 3):
    """Get segmentation from isolated galaxy images based on noise level cutoff.

    Input of `isolated_images` should be a single band.

    Args:
        isolate_images: Images of isolated galaxies. Shape only be in a single band.
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


def get_match_stats(
    pred_matches: List[np.ndarray], n_trues: np.ndarray, n_preds: np.ndarray
) -> tuple:
    """Return statistics on matches including tp, fp, t, p."""
    n = len(pred_matches)
    tp = np.zeros(len(n))
    fp = np.zeros(len(n))
    t = np.zeros(len(n))
    p = np.zeros(len(n))
    for ii in range(n):
        tp[ii] = np.sum(pred_matches[ii] >= 0)
        fp[ii] = np.sum(pred_matches[ii] == -1)
        t[ii] = n_trues[ii]
        p[ii] = n_preds[ii]
    return tp, fp, t, p


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
    return np.power((image1 - image2), 2).mean(axis=(-1, -2)).sqrt()


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


def psnr(image1: np.ndarray, image2: np.ndarray, **kwargs) -> np.ndarray:
    """Compute peak-signal-to-noise-ratio from skimage.

    Args:
        images1: Array of shape `NHW` containing `N` images each of
                size `NHW`.
        images2: Array of shape `NHW` containing `N` images each of
                size `NHW`.
        kwargs: Keyword arguments to be passed in to `peak_signal_noise_ratio` function
                within `skimage.metrics`.

    Returns:
        Returns PSNR between each corresponding iamge as an array of shape `N`.
    """
    n, h, w = image1.shape
    assert (n, h, w) == image2.shape
    psnr_ = np.zeros(n)
    for ii in range(n):
        psnr_[ii] = peak_signal_noise_ratio(image1[ii], image2[ii], **kwargs)
    return psnr_


def struct_sim(image1: np.ndarray, image2: np.ndarray, **kwargs) -> np.ndarray:
    """Compute structural similarity index from skimage.

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
    n, h, w = image1.shape
    assert (n, h, w) == image2.shape
    ssim = np.zeros(n)
    for ii in range(n):
        ssim[ii] = structural_similarity(image1[ii], image2[ii], **kwargs)
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
