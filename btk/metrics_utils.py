"""Atomic metric functions."""

from typing import List

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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


def iou(segmentation1: np.ndarray, segmentation2: np.ndarray) -> np.ndarray:
    """Calculates intersection-over-union (IoU) given two semgentation arrays.

    The segmentation arrays should each have values of 1 or 0s only.

    Args:
        segmentation1: Array of shape `NHW` containing `N` segmentation maps each of
                        size `HW`.
        segmentation2: Array of shape `NHW` containing `N` segmentation maps each of
                        size `HW`.

    Returns:
        Returns `iou` between each corresponding segmentation map as an array of shape `N`

    """
    seg1 = segmentation1.astype(bool)
    seg2 = segmentation2.astype(bool)
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


def match_stats(match_list: List[np.ndarray], n_preds: np.ndarray) -> tuple:
    """Return statistics on matches including tp, fp, t, p."""
    n = len(match_list)
    tp = np.zeros(len(n))
    fp = np.zeros(len(n))
    t = np.zeros(len(n))
    p = np.zeros(len(n))
    for ii in range(n):
        tp[ii] = np.sum(match_list[ii] >= 0)
        fp[ii] = np.sum(match_list[ii] == -1)
        t[ii] = len(match_list[ii])
        p[ii] = n_preds[ii]
    return tp, fp, t, p


def efficiency_matrix(tp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Returns efficiency matrices based on number of matched truth and predicted galaxies."""
    n = len(t)
    max_true_n_sources = max(t)
    max_pred_n_sources = max(tp)
    eff_mat = np.zeros((max_pred_n_sources + 1, max_true_n_sources + 1))
    for ii in range(n):
        eff_mat[tp[ii], t[ii]] += 1
    return eff_mat
