"""Module for measuring galaxy properties."""
from typing import Tuple

import galsim
import numpy as np
import sep
from galsim import GSObject


def _get_single_ksb_ellipticity(
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


def get_ksb_ellipticity(
    images: np.ndarray, psf: GSObject, pixel_scale: float, verbose=False
) -> np.ndarray:
    """Return ellipticities of both true and detected galaxies, assuming they are matched."""
    # psf is assumed to be the same for the entire batch and correspond to selected band.
    assert len(images.shape) == 4  # (batch_size, max_n_sources, H, W)
    batch_size, max_n_sources, _, _ = images.shape
    ellipticities = np.zeros((batch_size, max_n_sources, 2))
    for ii in range(batch_size):
        for jj in range(max_n_sources):
            if np.sum(images[ii, jj]) > 0:
                ellipticities[ii, jj] = _get_single_ksb_ellipticity(
                    images[ii, jj], psf, pixel_scale, verbose=verbose
                )
            else:
                ellipticities[ii, jj] = (np.nan, np.nan)
    return ellipticities


def get_blendedness(iso_image: np.ndarray):
    """Calculate blendedness given isolated images of each galaxy in a blend.

    Args:
        iso_image: Array of shape = (..., N, H, W) corresponding to images of the isolated
            galaxy you are calculating blendedness for.
    """
    assert iso_image.ndim >= 3
    num = np.sum(iso_image * iso_image, axis=(-1, -2))
    blend = np.sum(iso_image, axis=-3)[..., None, :, :]
    denom = np.sum(blend * iso_image, axis=(-1, -2))
    return 1 - num / denom


def get_snr(iso_image: np.ndarray, sky_level: float) -> float:
    """Calculate SNR of a set of isolated galaxies with same sky level.

    Args:
        iso_image: Array of shape = (..., H, W) corresponding to image of the isolated
            galaxy you are calculating SNR for.
        sky_level: Background level of all images. Images are assume to be
            background-substracted.
    """
    images = iso_image + sky_level
    return np.sqrt(np.sum(iso_image * iso_image / images, axis=(-1, -2)))


def _get_single_aperture_flux(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, radius: float, sky_level: float
) -> np.ndarray:
    """Utility function to measure flux using fixed circular aperture with sep.

    Args:
        image (np.array): Single iamge to measure flux on, with shape (H, W).
        x (np.array): x coordinates of the center of the aperture (in pixels).
        y (np.array): y coordinates of the center of the aperture (in pixels).
        sky_level (float): Background level of all images.
            Images are assume to be background substracted.
        radius (float): Radius of the aperture in pixels.
    """
    assert image.ndim == 2
    flux, _, _ = sep.sum_circle(image, x, y, radius, err=sky_level)
    return flux[0]


def get_aperture_fluxes(
    images: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float, sky_level: float
) -> np.ndarray:
    """Utility function to measure flux using fixed circular aperture with sep.

    Args:
        images (np.array): Images to measure flux on, with shape (B, H, W).
        xs (np.array): x coordinates of the center of the aperture (in pixels).
        ys (np.array): y coordinates of the center of the aperture (in pixels).
        sky_level (float): Background level of all images.
            Images are assume to be background substracted.
        radius (float): Radius of the aperture in pixels.
    """
    assert images.ndim == 3
    batch_size = images.shape[0]
    fluxes = np.zeros((batch_size, len(xs)))
    for ii in range(batch_size):
        fluxes[ii] = _get_single_aperture_flux(images[ii], xs[ii], ys[ii], radius, sky_level)
    return fluxes


def get_residual_images(iso_images: np.ndarray, blend_images: np.ndarray) -> np.ndarray:
    """Calculate residual images given isolated images of each galaxy in a blend.

    Args:
        iso_images: Array of shape = (B, N, H, W) corresponding to images of the isolated
            galaxies you are calculating residual images for.
        blend_images: Array of shape = (B, H, W) where B is the batch size. Contains noise.
    """
    except_one_images = np.sum(iso_images, axis=1)[:, None] - iso_images
    return blend_images[:, None] - except_one_images
