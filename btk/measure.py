"""Module for measuring galaxy properties from images."""

from typing import Tuple

import galsim
import numpy as np
import sep_pjw as sep
from galsim import GSObject


def _get_single_ksb_ellipticity(
    image: np.ndarray, centroid: np.ndarray, psf: GSObject, pixel_scale: float, verbose=False
) -> Tuple[float, float]:
    """Utility function to measure ellipticity using the KSB method.

    Args:
        image: Image of a single, isolated galaxy with shape (h, w).
        centroid: The centroid of the galaxy in the image, with shape (2,). Following the GalSim
            convention for offsets.
        psf: A galsim object containing the PSF of the single, isolated galaxy.
        pixel_scale: The pixel scale of the galaxy image.
        verbose: Whether to print errors if they happen when estimating ellipticity.

    Return:
        Tuple of (g1, g2) containing measured shapes.
    """
    psf_image = galsim.Image(image.shape[0], image.shape[1], scale=pixel_scale)
    gal_image = galsim.Image(image, scale=pixel_scale)
    psf_image = psf.drawImage(psf_image)
    pos = galsim.PositionD(centroid)

    res = galsim.hsm.EstimateShear(
        gal_image, psf_image, shear_est="KSB", strict=False, guess_centroid=pos
    )
    output = (res.corrected_g1, res.corrected_g2)
    if res.error_message != "":  # absorbs all (10, -10) and makes them np.nan
        output = (np.nan, np.nan)
        if verbose:
            print(
                f"Shear measurement error: '{res.error_message }'. \
                This error may happen for faint galaxies or inaccurate detections."
            )
    return output


def get_ksb_ellipticity(
    images: np.ndarray, centroids: np.ndarray, psf: GSObject, pixel_scale: float, verbose=False
) -> np.ndarray:
    """Calculate the KSB ellipticities of a batched array of isolated galaxy images.

    The galaxy images are assumed to all correspond to single band, and the input PSF is assumed
    to be the same for all images.

    If the shear measurement fails or the image is empty (no flux), then `np.nan` is returned for
    the corresponding ellipticity.

    Args:
        images: Array of batch isolated images with shape (batch_size, max_n_sources, h, w)
        centroids: An array of centers for each galaxy using the GalSim convention where the
            center of the lower-left pixel is (image.xmin, image.ymin). The shape of this array is
            (batch_size, max_n_sources, 2).
        psf: a GalSim GSObject containing the PSF common to all galaxies.
        pixel_scale: The pixel scale of the galaxy images.
        verbose: Whether an error message should be printed if the ellipticity measurement fails
            for any one of the galaxies.

    Returns:
        An array containing the measured ellipticities of shape (batch_size, max_n_sources, 2)
    """
    # psf is assumed to be the same for the entire batch and correspond to selected band.
    assert images.ndim == 4
    batch_size, max_n_sources, _, _ = images.shape
    ellipticities = np.zeros((batch_size, max_n_sources, 2))
    for ii in range(batch_size):
        for jj in range(max_n_sources):
            if np.sum(images[ii, jj]) > 0:
                ellipticities[ii, jj] = _get_single_ksb_ellipticity(
                    images[ii, jj], centroids[ii, jj], psf, pixel_scale, verbose=verbose
                )
            else:
                ellipticities[ii, jj] = (np.nan, np.nan)
    return ellipticities


def get_blendedness(iso_image: np.ndarray) -> np.ndarray:
    """Calculate blendedness given isolated images of each galaxy in a blend.

    Args:
        iso_image: Array of shape = (..., N, H, W) corresponding to images of isolated
            galaxies you are calculating blendedness for.

    Returns:
        Array of size (..., N) corresponding to blendedness values for each individual galaxy.
    """
    assert iso_image.ndim >= 3
    num = np.sum(iso_image * iso_image, axis=(-1, -2))
    blend = np.sum(iso_image, axis=-3)[..., None, :, :]
    denom = np.sum(blend * iso_image, axis=(-1, -2))
    return 1 - np.divide(num, denom, out=np.ones_like(num), where=(num != 0))


def get_snr(iso_image: np.ndarray, sky_level: float) -> np.ndarray:
    """Calculate SNR of a set of isolated galaxies with same sky level.

    Args:
        iso_image: Array of shape = (..., H, W) corresponding to image of the isolated
            galaxy you are calculating SNR for.
        sky_level: Background level of all images. Images are assume to be
            background-substracted.

    Returns:
        Array of size (...) corresponding to SNR values for each individual galaxy.
    """
    images = iso_image + sky_level
    return np.sqrt(np.sum(iso_image * iso_image / images, axis=(-1, -2)))


def _get_single_aperture_flux(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, radius: float, sky_level: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Utility function to measure flux using fixed circular aperture with sep.

    Args:
        image (np.array): Single iamge to measure flux on, with shape (H, W).
        x (np.array): x coordinates of the center of the aperture (in pixels).
        y (np.array): y coordinates of the center of the aperture (in pixels).
        sky_level (float): Background level of all images.
            Images are assume to be background substracted.
        radius (float): Radius of the aperture in pixels.

    Returns:
        Tuple of flux and fluxerr.
    """
    assert image.ndim == 2
    assert x.ndim == 1 and y.ndim == 1
    flux, fluxerr, _ = sep.sum_circle(image, x, y, radius, var=sky_level)
    return flux, fluxerr


def get_aperture_fluxes(
    images: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float, sky_level: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Utility function to measure flux using fixed circular aperture with sep.

    Args:
        images (np.array): Images to measure flux on, with shape (B, H, W).
        xs (np.array): x coordinates of the center of the aperture (in pixels).
        ys (np.array): y coordinates of the center of the aperture (in pixels).
        sky_level (float): Background level of all images.
            Images are assume to be background substracted.
        radius (float): Radius of the aperture in pixels.

    Returns:
        fluxes (np.array): Array of shape (B, N) corresponding to the measured aperture fluxes
            in each given position for each of the B batches.
        fluxerr (np.array): Array of same shape with corresponding flux errors.
    """
    assert images.ndim == 3
    assert xs.ndim == 2 and ys.ndim == 2
    batch_size, max_n_sources = xs.shape
    fluxes = np.zeros((batch_size, max_n_sources))
    fluxerrs = np.zeros((batch_size, max_n_sources))
    for ii in range(batch_size):
        n_sources = np.sum((xs[ii] > 0) & (ys[ii] > 0)).astype(int)
        flux, err = _get_single_aperture_flux(images[ii], xs[ii], ys[ii], radius, sky_level)
        fluxes[ii, :n_sources] = flux[:n_sources]
        fluxerrs[ii, :n_sources] = err[:n_sources]
    return fluxes, fluxerrs


def get_residual_images(iso_images: np.ndarray, blend_images: np.ndarray) -> np.ndarray:
    """Calculate residual images given isolated images of each galaxy in a blend.

    Args:
        iso_images: Array of shape = (B, N, H, W) corresponding to images of the isolated
            galaxies you are calculating residual images for.
        blend_images: Array of shape = (B, H, W) where B is the batch size. Contains noise.
    """
    except_one_images = np.sum(iso_images, axis=1)[:, None] - iso_images
    return blend_images[:, None] - except_one_images
