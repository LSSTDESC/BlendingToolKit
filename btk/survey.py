"""Contains information for surveys available in BTK."""
import os
import random as rd
from collections.abc import Callable

import astropy.wcs as WCS
import galcheat
import galsim
import numpy as np
from astropy import units as u
from astropy.io import fits


def get_surveys(names="LSST", psf_func: Callable = None):
    """Return specified surveys from galcheat extended to contain PSF information.

    Args:
        names (str or list): A single str specifying a survey from conf/surveys or a list with
            multiple survey names.
        psf_func (function): Python function which takes in two arguments: `survey` and `filter`
            that returns a PSF as a galsim object or as a callable with no arguments.
            If `None`, the default PSF for the specified survey will be used in each band.

    Returns:
        galcheat.survey.Survey object or list of such objects.
    """
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list):
        raise TypeError("Argument 'names' of `get_surveys` should be a str or list.")

    # add PSF to filters
    surveys = []
    for survey_name in names:
        survey = galcheat.get_survey(survey_name)
        for band in survey.available_filters:
            filtr = survey.get_filter(band)
            if psf_func is None:
                psf = get_default_psf_with_galcheat_info(survey, filtr)
            else:
                psf = psf_func(survey, filtr)
            filtr.psf = psf
        surveys.append(survey)

    if len(surveys) == 1:
        surveys = surveys[0]
    return surveys


def get_default_psf_with_galcheat_info(
    survey: galcheat.survey.Survey, filtr: galcheat.filter.Filter
):
    """Return the default PSF model as a galsim object based on galcheat survey parameters.

    Args:
        survey (galcheat.survey.Survey): Survey object from galcheat.
        filtr (galcheat.filter.Filter): Filter object from galcheat.

    Returns:
        btk.survey.Survey object or list of such objects.
    """
    return get_default_psf(
        survey.mirror_diameter.to(u.m).value,
        survey.effective_area.to(u.m**2).value,
        filtr.psf_fwhm.to(u.arcsec).value,
        filt_wavelength=filtr.effective_wavelength.to(u.Angstrom).value,
        atmospheric_model="Kolmogorov",
    )


def get_default_psf(
    mirror_diameter,
    effective_area,
    fwhm,
    filt_wavelength,
    atmospheric_model="Kolmogorov",
):
    """Defines a synthetic galsim PSF model.

    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        mirror_diameter (float): in meters [m]
        effective_area (float): effective total light collecting area in square meters [m2]
        filt_wavelength (string): filter wavelength in nanometers [nm]
        fwhm (float): fwhm of the atmospheric component in arcseconds. [arcsec]
        atmospheric_model (string): type of atmospheric model. Current options:
            ['Kolmogorov', 'Moffat'].

    Returns:
        psf_model: galsim psf model
    """
    # define atmospheric psf
    if atmospheric_model == "Kolmogorov":
        atmospheric_psf_model = galsim.Kolmogorov(fwhm=fwhm)
    elif atmospheric_model == "Moffat":
        atmospheric_psf_model = galsim.Moffat(2, fwhm=fwhm)
    elif atmospheric_model is None:
        atmospheric_psf_model = None
    else:
        raise NotImplementedError(
            f"The atmospheric model request '{atmospheric_model}' is incorrect or not implemented."
        )

    # define optical psf if available
    if mirror_diameter > 0:
        mirror_area = np.pi * (0.5 * mirror_diameter) ** 2
        area_ratio = effective_area / mirror_area
        if area_ratio <= 0 or area_ratio > 1:
            raise RuntimeError("Incompatible effective-area and mirror-diameter values.")
        obscuration_fraction = np.sqrt(1 - area_ratio)
        lambda_over_diameter = 3600 * np.degrees(1e-10 * filt_wavelength / mirror_diameter)
        optical_psf_model = galsim.Airy(
            lam_over_diam=lambda_over_diameter, obscuration=obscuration_fraction
        )
    else:
        optical_psf_model = None

    # define the psf model according to the components we have
    if isinstance(atmospheric_psf_model, galsim.GSObject) and isinstance(
        optical_psf_model, galsim.GSObject
    ):
        psf_model = galsim.Convolve(atmospheric_psf_model, optical_psf_model)
    elif isinstance(atmospheric_psf_model, galsim.GSObject) and optical_psf_model is None:
        psf_model = atmospheric_psf_model
    elif atmospheric_psf_model is None and isinstance(optical_psf_model, galsim.GSObject):
        psf_model = optical_psf_model
    elif atmospheric_psf_model is None and optical_psf_model is None:
        raise RuntimeError("Neither the atmospheric nor the optical PSF components are defined.")

    return psf_model.withFlux(1.0)


def get_psf_from_file(psf_dir, survey):
    """Generates a custom PSF galsim model from FITS file(s).

    Args:
        psf_dir (string): directory where the PSF FITS files are
        survey (btk Survey): BTK Survey object

    Returns:
        galsim PSF model
    """
    psf_files = os.listdir(psf_dir)
    if len(psf_files) > 1:
        psf_file = rd.choice(psf_files)
    elif len(psf_files) == 1:
        psf_file = psf_files[0]
    else:
        raise RuntimeError(f"No psf files found in '{psf_dir}'.")
    psf_array = fits.getdata(psf_dir + "/" + psf_file)
    psf_model = galsim.InterpolatedImage(
        galsim.Image(psf_array), scale=survey.pixel_scale.value
    ).withFlux(1.0)

    return psf_model


def make_wcs(pixel_scale, shape, center_pix=None, center_sky=None, projection="TAN"):
    """Creates WCS for an image.

    The default (`center_pix=None` AND `center_sky=None`) is that the center of the image in
    pixels [(s + 1) / 2, (s + 1) / 2] corresponds to (ra, dec) = [0, 0].

    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image in pixels.
        center_pix (tuple): tuple representing the center of the image in pixels
        center_sky (tuple): tuple representing the center of the image in sky coordinates
                     (RA,DEC) in arcseconds.
        projection(str): projection type, default to TAN. A list of available
                            types can be found in astropy.wcs documentation

    Returns:
        astropy WCS
    """
    if center_pix is None:
        center_pix = [(s + 1) / 2 for s in shape]
    if center_sky is None:
        center_sky = [0 for _ in range(2)]
    w = WCS.WCS(naxis=2)
    w.wcs.ctype = ["RA---" + projection, "DEC--" + projection]
    w.wcs.crpix = center_pix
    w.wcs.cdelt = [pixel_scale / 3600 for _ in range(2)]
    w.wcs.crval = [c / 3600 for c in center_sky]
    w.array_shape = shape
    return w
