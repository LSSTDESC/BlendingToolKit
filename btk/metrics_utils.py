"""Atomic metric functions."""

from typing import Callable, List, Tuple

import galsim
import numpy as np
import sep
from astropy.table import Table
from galcheat.utilities import mean_sky_level
from galsim import GSObject
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def meas_ksb_ellipticity(
    image: np.ndarray, psf: GSObject, pixel_scale: float, verbose=False
) -> Tuple[float, float]:
    """Utility function to measure ellipticity using the KSB method.

    Args:
        image: Image of a single, isolated galaxy with shape (H, W).
        psf: A galsim object containing the PSF of the single, isolated galaxy.
        pixel_scale: The pixel scale of the galaxy image.
        verbose: Whether to print errors if they happen when estimating ellipticity.

    Return:
        Tuple of (g1, g2) containing measured shapes.
    """
    psf_image = galsim.Image(image.shape[0], image.shape[1], scale=pixel_scale)
    psf_image = psf.drawImage(psf_image)
    gal_image = galsim.Image(image, scale=pixel_scale)

    res = galsim.hsm.EstimateShear(gal_image, psf_image, shear_est="KSB", strict=False)
    output = (res.corrected_g1, res.corrected_g2)
    if res.error_message != "" and verbose:
        print(
            f"Shear measurement error: '{res.error_message }'. \
            This error may happen for faint galaxies or inaccurate detections."
        )
        output = (np.nan, np.nan)
    return output


def get_blendedness(iso_image: np.ndarray, blend_iso_images: np.ndarray):
    """Calculate blendedness given isolated images of each galaxy in a blend.

    Args:
        iso_image: Array of shape = (H,W) corresponding to image of the isolated
            galaxy you are calculating blendedness for.
        blend_iso_images: Array of shape = (N, H, W) where N is the number of galaxies
            in the blend and each image of this array corresponds to an isolated galaxy that is
            part of the blend (includes `iso_image`).
    """
    num = np.sum(iso_image * iso_image)
    denom = np.sum(np.sum(blend_iso_images, axis=0) * iso_image)
    return 1 - num / denom


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
    psnr = np.zeros(n)
    for ii in range(n):
        psnr[ii] = peak_signal_noise_ratio(image1[ii], image2[ii], **kwargs)
    return psnr


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


def distance_center(cat1: Table, cat2: Table) -> np.ndarray:
    """Computes the euclidean distance between the two galaxies given as arguments.

    Args:
        cat1: Catalog containing at least columns `x_peak` and `y_peak`.
        cat2: Catalog containing at least columns `x_peak` and `y_peak`.

    Returns:
        Distance between corresponding galaxies in each catalog as an np.ndarray.
    """
    x_peak1, y_peak1 = cat1["x_peak"].data, cat1["y_peak"].data
    x_peak2, y_peak2 = cat2["x_peak"].data, cat2["y_peak"].data
    return np.hypot(x_peak1 - x_peak2, y_peak1 - y_peak2)


def get_id_matches(truth: Table, pred: Table) -> Tuple[np.ndarray, np.ndarray]:
    """Assume catalogs are already matched one-to-one."""
    dist = distance_center(truth, pred)
    return list(range(len(truth))), dist


def get_least_dist_matches(
    truth: Table,
    pred: Table,
    f_distance: Callable = distance_center,
    dist_thresh: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Uses the Hungarian algorithm to find optimal matching between detections and true objects.

    The optimal matching is computed based on the following optimization problem:

    .. math::

        \sum_{i} \sum_{j} C_{i,j} X_{i,j}

    where, in the BTK context, :math:`C_{ij}` is the cost function between matching true object
    :math:`i` with detected object :math:`j` computed as the L2 distance between the two objects,
    and :math:`X_{i,j}` is an indicator function over the matches.

    Based on this implementation in scipy:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

    Args:
        truth: Table with entries corresponding to
            the true object parameter values in one blend.
        pred: Table with entries corresponding
            to output of measurement algorithm in one blend.
        f_distance: Function used to compute the distance between true and detected
            galaxies. Takes as arguments the entries corresponding to the two galaxies.
            By default the distance is the euclidean distance from center to center.
        dist_thresh: Maximum distance for matching a detected and a
                true galaxy in pixels.

    Returns:
        Tuple containing two elements:
            - "matches": Index of row in `pred` table corresponding to
                matched detected object. If no match, value is -1.
            - "dist_m": distance between true object and matched object or 0 if not matched.

    """
    if "x_peak" not in truth.colnames:
        raise KeyError("True table has no column x_peak")
    if "y_peak" not in truth.colnames:
        raise KeyError("True table has no column y_peak")
    if "x_peak" not in pred.colnames:
        raise KeyError("Detection table has no column x_peak")
    if "y_peak" not in pred.colnames:
        raise KeyError("Detection table has no column y_peak")

    # dist[i][j] = distance between true object i and detected object j.
    dist = np.zeros((len(truth), len(pred)))
    for i, true_gal in enumerate(truth):
        for j, detected_gal in enumerate(pred):
            dist[i][j] = f_distance(true_gal, detected_gal)

    # solve optimization problem.
    # true_table[true_indx[i]] is matched with detected_table[detected_indx[i]]
    # len(true_indx) = len(detect_indx) = min(len(true_table), len(detected_table))
    true_indx, detected_indx = linear_sum_assignment(dist)

    # for each true galaxy i, match_indx[i] is the index of detected_table matched to that true
    # galaxy or -1 if there is no match.
    match_indx = [-1] * len(truth)
    dist_m = [0.0] * len(truth)
    for i, indx in enumerate(true_indx):
        if dist[indx][detected_indx[i]] <= dist_thresh:
            match_indx[indx] = detected_indx[i]
            dist_m[indx] = dist[indx][detected_indx[i]]

    return match_indx, dist_m


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


def meas_fixed_aperture(image, additional_params):
    """Utility function to measure flux using fixed circular aperture with sep.

    Args:
        image (np.array): Image of a single, isolated galaxy with shape (H, W).
        additional_params (dict): Containing keys 'psf', 'survey' and 'meas_band_num'.
                                  The psf should be a Galsim PSF model, the survey a btk Survey
                                  and meas_band_num an integer indicating the band in which the
                                  measurement is done.
    """
    meas_band_num = additional_params["meas_band_num"]
    band = additional_params["survey"].available_filters[meas_band_num]
    filt = additional_params["survey"].get_filter(band)
    sky_level = np.sqrt(mean_sky_level(additional_params["survey"], filt).to_value("electron"))
    pixel_scale = additional_params["survey"].pixel_scale.to_value("arcsec")
    verbose = additional_params["verbose"]
    catalog = sep.extract(image[meas_band_num], 1.5, err=sky_level)
    if len(catalog) != 1 and verbose:
        print(f"{len(catalog)} where detected when measuring flux.")

    flux, fluxerr, flag = sep.sum_circle(
        image[meas_band_num],
        catalog["x"],
        catalog["y"],
        filt.psf_fwhm.to_value("arcsec") / pixel_scale,
        err=sky_level,
    )
    result = [flux[0], fluxerr[0]]
    return result
