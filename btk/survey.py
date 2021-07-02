"""Contains information for surveys available in BTK."""
import os
import random as rd
from collections import namedtuple
from typing import Iterable

import astropy.wcs as WCS
import galsim
import numpy as np
from astropy.io import fits
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf


Survey = namedtuple(
    "Survey",
    [
        "name",
        "pixel_scale",  # arcseconds per pixel
        "effective_area",  # Effective total light collecting area in square meters [m2]
        "mirror_diameter",  # in meters [m]
        "airmass",  # Optical path length through atmosphere relative to zenith path length.
        "filters",
        "zeropoint_airmass",
    ],
)

Survey.__doc__ = """
Class containing the informations relative to a survey.

Args:
    name (str) : Name of the survey
    pixel_scale (float) : Pixel scale of the survey, in arcseconds per pixel
    effective_area (float) : Effective total light collecting area, in square meters
    mirror_diameter (float) : Diameter of the primary mirror, in meters
    airmass (float) : Optical path length through atmosphere relative to zenith path length
    filters (list) : List of Filter objects corresponding to the filters of this survey
    zeropoint_airmass (float)"""

Filter = namedtuple(
    "Filter",
    [
        "name",
        "psf",  # galsim psf model or function to generate it
        "sky_brightness",  # mags/sq.arcsec
        "exp_time",  # in seconds [s]
        "zeropoint",  # in electrons per second at 24th magnitude.
        "extinction",  # Exponential extinction coefficient for atmospheric absorption.
    ],
)

Filter.__doc__ = """
Class containing the informations relative to a filter (for a specific survey).

Args:
    name (str) : Name of the filter
    psf : Contains the PSF information, either as a Galsim object,
          or as a function returning a Galsim object
    sky_brightness (float) : Sky brightness, in mags/sq.arcsec
    exp_time (int) : Total exposition time, in seconds
    zeropoint (float) : Simulated camera zero point in electrons per second at 24th magnitude
    extinction (float) : Exponential extinction coefficient for atmospheric absorption"""


def _get_survey_from_cfg(survey_conf: OmegaConf):
    """Creates the corresponding `btk.survey.Survey` object using the information from config file.

    Args:
        cfg (OmegaConf): Hydra configuration object corresponding to a single survey.

    Returns:
        btk.survey.Survey object.
    """
    OmegaConf.resolve(survey_conf)  # in-place

    survey_dict = dict(survey_conf)
    filters_dict = dict(survey_dict.pop("filters"))
    filter_names = list(filters_dict.keys())
    for key in filter_names:
        filter_dict = dict(filters_dict.pop(key))  # need to make it a dict not DictConfig
        psf_dict = dict(filter_dict.pop("psf"))
        if "type" not in psf_dict:
            raise AttributeError(
                f"Your configuration for the PSF in filter {key} in survey {survey_dict['name']}"
                f"does not have a 'type' field which is required."
            )
        if psf_dict["type"] == "default":
            psf = get_psf(**psf_dict["params"])
        elif psf_dict["type"] == "galsim":
            galsim_dict = dict(psf=psf_dict["params"])
            psf = galsim.config.BuildGSObject(galsim_dict, "psf")
        else:
            raise NotImplementedError(
                f"The 'type' specified for the PSF in filter {key} in "
                f"survey {survey_dict['name']} is not implemented."
            )

        filter_dict["psf"] = psf
        filters_dict[key] = filter_dict
    filters = [Filter(**filters_dict[key]) for key in filters_dict]
    survey = Survey(**survey_dict, filters=filters)
    return survey


def get_surveys(names="Rubin", overrides: Iterable = ()):
    """Return specified surveys as `btk.survey.Survey` objects.

    NOTE: The surveys currently implemented correspond to config files inside `conf/surveys`. See
        the documentation for how to add your own surveys via custom config files.

    Args:
        names (str or list): A single str specifying a survey from conf/surveys or a list with
            multiple survey names.
        overrides (Iterable): List or tuple containg overrides for the survey config files. An
            example element of overrides could be 'surveys.Rubin.airmass=1.1', i.e. what you would
            pass into the CLI in order to customize the surveys used (here specified by `names`).

    Returns:
        btk.survey.Survey object or list of such objects.
    """
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list):
        raise TypeError("Argument 'names' of `get_surveys` should be a str or list.")
    overrides = [f"surveys={names}", *overrides]
    surveys = []
    with initialize(config_path="../conf"):
        cfg = compose("config", overrides=overrides)
    for survey_name in cfg.surveys:
        survey_conf = cfg.surveys[survey_name]
        surveys.append(_get_survey_from_cfg(survey_conf))
    if len(surveys) == 1:
        return surveys[0]
    return surveys


def get_psf(
    mirror_diameter,
    effective_area,
    filt_wavelength,
    fwhm,
    atmospheric_model="Kolmogorov",
):
    """Defines a synthetic galsim PSF model.

    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        mirror_diameter (float): in meters [m]
        effective_area (float): effective total light collecting area in square meters [m2]
        filt_wavelength (string): filter wavelength
        fwhm (float): fwhm of the atmospheric component
        atmospheric_model (string): type of atmospheric model

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
        survey (btk Survey) : BTK Survey object

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
        galsim.Image(psf_array), scale=survey.pixel_scale
    ).withFlux(1.0)

    return psf_model


def get_flux(ab_magnitude, filt, survey):
    """Convert source magnitude to flux.

    The calculation includes the effects of atmospheric extinction.
    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        ab_magnitude(float): AB magnitude of source.
        filt (btk.survey.Filter) : BTK Filter object
        survey (btk.survey.Survey) : BTK Survey object

    Returns:
        Flux in detected electrons.
    """
    mag = ab_magnitude + filt.extinction * (survey.airmass - survey.zeropoint_airmass)
    return filt.exp_time * filt.zeropoint * 10 ** (-0.4 * (mag - 24))


def get_mean_sky_level(survey, filt):
    """Computes the mean sky level given to Galsim for noise generation.

    Args:
        survey (btk.survey.Survey) : BTK Survey object
        filt (btk.survey.Filter) : BTK Filter object

    Returns:
        Corresponding mean sky level
    """
    return get_flux(filt.sky_brightness, filt, survey) * survey.pixel_scale ** 2


def make_wcs(pixel_scale, shape, center_pix=None, center_sky=None, projection="TAN"):
    """Creates WCS for an image.

    The default (`center_pix=None` AND `center_sky=None`) is that the center of the image in
    pixels [(s + 1) / 2, (s + 1) / 2] corresponds to (ra, dec) = [0, 0].

    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image in pixels.
        center_pix (tuple) : tuple representing the center of the image in pixels
        center_sky (tuple) : tuple representing the center of the image in sky coordinates
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
