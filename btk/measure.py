"""Module for measuring galaxy properties."""
from typing import Tuple

import galsim
import numpy as np
import sep
from galsim import GSObject


def get_single_ksb_ellipticity(
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
    iso_images: np.ndarray, psf: GSObject, pixel_scale: float, verbose=False
) -> np.ndarray:
    """Return ellipticities of both true and detected galaxies, assuming they are matched."""
    # psf is assumed to be the same for the entire batch and correspond to selected band.
    assert len(iso_images.shape) == 4  # (batch_size, max_n_sources, H, W)
    batch_size, max_n_sources, _, _ = iso_images.shape
    ellipticities = np.zeros((batch_size, max_n_sources, 2))
    for ii in range(batch_size):
        for jj in range(max_n_sources):
            if np.sum(iso_images[ii, jj]) > 0:
                ellipticities[ii, jj] = get_single_ksb_ellipticity(
                    iso_images[ii, jj], psf, pixel_scale, verbose=verbose
                )
            else:
                ellipticities[ii, jj] = (np.nan, np.nan)
    return ellipticities


def get_blendedness(iso_image: np.ndarray, blend_iso_images: np.ndarray):
    """Calculate blendedness given isolated images of each galaxy in a blend.

    Args:
        iso_image: Array of shape = (..., H, W) corresponding to image of the isolated
            galaxy you are calculating blendedness for.
        blend_iso_images: Array of shape = (..., H, W) where N is the number of galaxies
            in the blend and each image of this array corresponds to an isolated galaxy that is
            part of the blend (includes `iso_image`).
    """
    num = np.sum(iso_image * iso_image, axis=(1, 2))
    blend = np.sum(blend_iso_images, axis=1)[:, None]
    denom = np.sum(blend * iso_image, axis=(1, 2))
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
    err = np.sqrt(sky_level)
    return np.sqrt(np.sum(images * images, axis=(1, 2))) / err


def get_residual_images(iso_images: np.ndarray, blend_images: np.ndarray):
    """Calculate residual images given isolated images of each galaxy in a blend.

    Args:
        iso_images: Array of shape = (B, N, H, W) corresponding to images of the isolated
            galaxies you are calculating residual images for.
        blends: Array of shape = (B, H, W) where B is the batch size. Contains noise.
    """
    except_one_images = np.sum(iso_images, axis=1)[:, None] - iso_images
    return blend_images[:, None] - except_one_images


def get_aperture_flux(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, sky_level: float, radius: float
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

    flux, _, _ = sep.sum_circle(image, x, y, radius, err=sky_level)
    return flux[0]


def get_aperture_fluxes(
    images: np.ndarray, xs: np.ndarray, ys: np.ndarray, sky_level: float = 0, radius: float = 3
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
    batch_size = images.shape[0]
    fluxes = np.zeros((batch_size, len(xs)))
    for ii in range(batch_size):
        fluxes[ii] = get_aperture_flux(images[ii], xs[ii], ys[ii], sky_level, radius)
    return fluxes
