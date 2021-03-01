"""Contains information for surveys available in BTK."""
from collections import namedtuple

import astropy.wcs as WCS
import random as rd
import numpy as np
import galsim
from collections import namedtuple
from astropy.io import fits


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


def get_psf(
    mirror_diameter,
    effective_area,
    filt_wavelength,
    fwhm,
    atmospheric_model="Kolmogorov",
):
    """Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)
    Defines a synthetic galsim PSF model
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
            raise RuntimeError(
                "Incompatible effective-area and mirror-diameter values."
            )
        obscuration_fraction = np.sqrt(1 - area_ratio)
        lambda_over_diameter = 3600 * np.degrees(
            1e-10 * filt_wavelength / mirror_diameter
        )
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
    elif (
        isinstance(atmospheric_psf_model, galsim.GSObject) and optical_psf_model is None
    ):
        psf_model = atmospheric_psf_model
    elif atmospheric_psf_model is None and isinstance(
        optical_psf_model, galsim.GSObject
    ):
        psf_model = optical_psf_model
    elif atmospheric_psf_model is None and optical_psf_model is None:
        raise RuntimeError(
            f"Neither the atmospheric nor the optical PSF components are defined."
        )

    return psf_model.withFlux(1.0)


def get_psf_from_file(psf_dir):
    """Generates a custom PSF galsim model from FITS file(s)
    Args:
        psf_dir (string): directory where the PSF FITS files are
    Returns:
        psf_model: galsim PSF model
    """

    psf_files = os.listdir(psf_dir)
    if len(psf_files) > 1:
        psf_file = rd.choice(psf_files)
    elif len(psf_files) == 1:
        psf_file = psf_files[0]
    else:
        raise RuntimeError(f"Not psf files found in '{psf_dir}'.")
    psf_array = fits.getdata(psf_file)
    psf_model = galsim.InterpolatedImage(
        galsim.Image(psf_array), scale=survey.pixel_scale
    ).withFlux(1.0)

    return psf_model


# Central wavelengths in Angstroms for each filter band, calculated from the
# baseline total filter throughputs tabulated at
# http://dev.lsstcorp.org/cgit/LSST/sims/throughputs.git/snapshot/throughputs-1.2.tar.gz
_central_wavelength = {
    "u": 3592.13,
    "g": 4789.98,
    "r": 6199.52,
    "i": 7528.51,
    "z": 8689.83,
    "y": 9674.05,
    "VIS": 7135.0,
    "f814w": 5000.0,  # TODO: placeholder
}

# https://sci.esa.int/documents/33859/36320/1567253682555-Euclid_presentation_Paris_1Dec2009.pdf
# http://www.mssl.ucl.ac.uk/~smn2/instrument.html
# area in square meters after 13% obscuration as in: https://arxiv.org/pdf/1608.08603.pdf
# exposure time: 4 exposures combined as in Cropper et al. 2018
# sky brightness: http://www.mssl.ucl.ac.uk/~smn2/instrument.html
# extinction: No atmosphere
Euclid = Survey(
    name="Euclid",
    pixel_scale=0.101,
    effective_area=1.15,
    mirror_diameter=1.3,
    airmass=1.0,
    zeropoint_airmass=1.0,
    filters=[
        Filter(
            name="VIS",
            psf=get_psf(
                mirror_diameter=1.3,
                effective_area=1.15,
                filt_wavelength=_central_wavelength["VIS"],
                fwhm=0.17,
            ),
            sky_brightness=22.9207,
            exp_time=2260,
            zeropoint=6.85,
            extinction=0.05,
        )
    ],
)

# Source https://hst-docs.stsci.edu/display/WFC3IHB/6.6+UVIS+Optical+Performance#id-6
# .6UVISOpticalPerformance-6.6.1 800nm
HST = Survey(
    name="HST",
    pixel_scale=0.06,
    effective_area=1.0,  # TODO: placeholder
    mirror_diameter=2.4,  # TODO: placeholder
    airmass=1.0,  # TODO: double-check
    zeropoint_airmass=1.0,  # TODO: double-check
    filters=[
        Filter(
            name="f814w",
            psf=get_psf(
                mirror_diameter=2.4,
                effective_area=1.0,
                filt_wavelength=_central_wavelength["f814w"],
                fwhm=0.074,
            ),
            sky_brightness=22,
            exp_time=3000,
            zeropoint=20,
            extinction=0,  # TODO: double-check
        )
    ],
)

# https://hsc-release.mtk.nao.ac.jp/doc/ deep+udeep
HSC = Survey(
    name="HSC",
    pixel_scale=0.167,
    effective_area=52.81,
    mirror_diameter=8.2,
    airmass=1.0,
    zeropoint_airmass=1.2,
    filters=[
        Filter(
            name="g",
            psf=get_psf(
                mirror_diameter=8.2,
                effective_area=52.81,
                filt_wavelength=_central_wavelength["g"],
                fwhm=0.72,
            ),
            sky_brightness=21.4,
            exp_time=600,
            zeropoint=91.11,
            extinction=0.13,
        ),
        Filter(
            name="r",
            psf=get_psf(
                mirror_diameter=8.2,
                effective_area=52.81,
                filt_wavelength=_central_wavelength["r"],
                fwhm=0.67,
            ),
            sky_brightness=20.6,
            exp_time=600,
            zeropoint=87.74,
            extinction=0.11,
        ),
        Filter(
            name="i",
            psf=get_psf(
                mirror_diameter=8.2,
                effective_area=52.81,
                filt_wavelength=_central_wavelength["i"],
                fwhm=0.56,
            ),
            sky_brightness=19.7,
            exp_time=1200,
            zeropoint=69.80,
            extinction=0.07,
        ),
        Filter(
            name="y",
            psf=get_psf(
                mirror_diameter=8.2,
                effective_area=52.81,
                filt_wavelength=_central_wavelength["y"],
                fwhm=0.64,
            ),
            sky_brightness=18.3,
            exp_time=1200,
            zeropoint=29.56,
            extinction=0.05,
        ),
        Filter(
            name="z",
            psf=get_psf(
                mirror_diameter=8.2,
                effective_area=52.81,
                filt_wavelength=_central_wavelength["z"],
                fwhm=0.64,
            ),
            sky_brightness=17.9,
            exp_time=1200,
            zeropoint=21.53,
            extinction=0.05,
        ),
    ],
)


# https://www.lsst.org/about/camera/features
Rubin = Survey(
    "LSST",
    pixel_scale=0.2,
    effective_area=32.4,
    mirror_diameter=8.36,
    airmass=1.2,
    zeropoint_airmass=1.2,
    filters=[
        Filter(
            name="y",
            psf=get_psf(
                mirror_diameter=8.36,
                effective_area=32.4,
                filt_wavelength=_central_wavelength["y"],
                fwhm=0.703,
            ),
            sky_brightness=18.6,
            exp_time=4800,
            zeropoint=10.58,
            extinction=0.138,
        ),
        Filter(
            name="z",
            psf=get_psf(
                mirror_diameter=8.36,
                effective_area=32.4,
                filt_wavelength=_central_wavelength["z"],
                fwhm=0.725,
            ),
            sky_brightness=19.6,
            exp_time=4800,
            zeropoint=22.68,
            extinction=0.043,
        ),
        Filter(
            name="i",
            psf=get_psf(
                mirror_diameter=8.36,
                effective_area=32.4,
                filt_wavelength=_central_wavelength["i"],
                fwhm=0.748,
            ),
            sky_brightness=20.5,
            exp_time=5520,
            zeropoint=32.36,
            extinction=0.07,
        ),
        Filter(
            name="r",
            psf=get_psf(
                mirror_diameter=8.36,
                effective_area=32.4,
                filt_wavelength=_central_wavelength["r"],
                fwhm=0.781,
            ),
            sky_brightness=21.2,
            exp_time=5520,
            zeropoint=43.70,
            extinction=0.10,
        ),
        Filter(
            name="g",
            psf=get_psf(
                mirror_diameter=8.36,
                effective_area=32.4,
                filt_wavelength=_central_wavelength["g"],
                fwhm=0.814,
            ),
            sky_brightness=22.3,
            exp_time=2400,
            zeropoint=50.70,
            extinction=0.163,
        ),
        Filter(
            name="u",
            psf=get_psf(
                mirror_diameter=8.36,
                effective_area=32.4,
                filt_wavelength=_central_wavelength["u"],
                fwhm=0.859,
            ),
            sky_brightness=22.9,
            exp_time=1680,
            zeropoint=9.16,
            extinction=0.451,
        ),
    ],
)

# http://www.ctio.noao.edu/noao/content/Basic-Optical-Parameters
# http://www.ctio.noao.edu/noao/content/DECam-What
# http://www.darkenergysurvey.org/survey/des-description.pdf
# skybrightness from http://www.ctio.noao.edu/noao/node/1218
# extinction from https://arxiv.org/pdf/1701.00502.pdf table 6
# fwhm values from https://arxiv.org/pdf/1407.3801.pdf
DES = Survey(
    name="DES",
    pixel_scale=0.263,
    effective_area=10.014,
    mirror_diameter=3.934,
    airmass=1.0,
    zeropoint_airmass=1.3,
    filters=[
        Filter(
            name="i",
            psf=get_psf(
                mirror_diameter=3.934,
                effective_area=10.014,
                filt_wavelength=_central_wavelength["i"],
                fwhm=0.96,
            ),
            sky_brightness=20.5,
            exp_time=1000,
            zeropoint=13.94,
            extinction=0.05,
        ),
        Filter(
            name="r",
            psf=get_psf(
                mirror_diameter=3.934,
                effective_area=10.014,
                filt_wavelength=_central_wavelength["r"],
                fwhm=1.03,
            ),
            sky_brightness=21.4,
            exp_time=800,
            zeropoint=15.65,
            extinction=0.09,
        ),
        Filter(
            name="g",
            psf=get_psf(
                mirror_diameter=3.934,
                effective_area=10.014,
                filt_wavelength=_central_wavelength["g"],
                fwhm=1.24,
            ),
            sky_brightness=22.3,
            exp_time=800,
            zeropoint=12.29,
            extinction=0.17,
        ),
        Filter(
            name="z",
            psf=get_psf(
                mirror_diameter=3.934,
                effective_area=10.014,
                filt_wavelength=_central_wavelength["z"],
                fwhm=1.12,
            ),
            sky_brightness=18.7,
            exp_time=800,
            zeropoint=10.81,
            extinction=0.06,
        ),
    ],
)

# http://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html
# http://www.cfht.hawaii.edu/Instruments/ObservatoryManual/om-focplndat.gif
# Calculating zeropoints with:
# http://www1.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/CFHTLS-SG/docs/extra/filters.html
CFHT = Survey(
    name="CFHT",
    pixel_scale=0.185,
    effective_area=8.022,
    mirror_diameter=3.592,
    airmass=1.0,  # TODO: double-check
    zeropoint_airmass=1.0,  # TODO: double-check
    filters=[
        Filter(
            name="i",
            psf=get_psf(
                mirror_diameter=3.592,
                effective_area=8.022,
                filt_wavelength=_central_wavelength["i"],
                fwhm=0.64,
            ),
            sky_brightness=20.3,
            exp_time=4300,
            zeropoint=8.46,
            extinction=0.07,
        ),
        Filter(
            name="r",
            psf=get_psf(
                mirror_diameter=3.592,
                effective_area=8.022,
                filt_wavelength=_central_wavelength["r"],
                fwhm=0.71,
            ),
            sky_brightness=20.8,
            exp_time=2000,
            zeropoint=10.72,
            extinction=0.10,
        ),
    ],
)


def get_flux(ab_magnitude, filt, survey):
    """Convert source magnitude to flux.
    The calculation includes the effects of atmospheric extinction.
    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        ab_magnitude(float): AB magnitude of source.
    Returns:
        float: Flux in detected electrons.
    """
    mag = ab_magnitude + filt.extinction * (survey.airmass - survey.zeropoint_airmass)
    return filt.exp_time * filt.zeropoint * 10 ** (-0.4 * (mag - 24))


def get_mean_sky_level(survey, filt):
    return get_flux(filt.sky_brightness, filt, survey) * survey.pixel_scale ** 2


def make_wcs(pixel_scale, shape, center_pix=None, center_sky=None, projection="TAN"):
    """Creates WCS for an image.
    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image in pixels.
        center_pix: tuple representing the center of the image in pixels
        center_sky: tuple representing the center of the image in sky coordinates
                     (RA,DEC) in arcseconds.
        projection(str): projection type, default to TAN. A list of available
                            types can be found in astropy.wcs documentation
    Returns:
        wcs: WCS
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
