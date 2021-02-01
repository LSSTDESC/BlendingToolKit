import astropy.wcs as WCS
import numpy as np
import galsim
from collections import namedtuple

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
        "atmospheric_psf_fwhm",  # in arcseconds.
        "sky_brightness",  # mags/sq.arcsec
        "exp_time",  # in seconds [s]
        "zeropoint",  # in electrons per second at 24th magnitude.
        "extinction",  # Exponential exctinction coefficient for atmospheric absorption.
    ],
)

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
            atmospheric_psf_fwhm=0.17,
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
    mirror_diameter=1.0,  # TODO: placeholder
    airmass=1.0,  # TODO: double-check
    zeropoint_airmass=1.0,  # TODO: double-check
    filters=[
        Filter(
            name="f814w",
            atmospheric_psf_fwhm=0.074,
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
            atmospheric_psf_fwhm=0.72,
            sky_brightness=21.4,
            exp_time=600,
            zeropoint=91.11,
            extinction=0.13,
        ),
        Filter(
            name="r",
            atmospheric_psf_fwhm=0.67,
            sky_brightness=20.6,
            exp_time=600,
            zeropoint=87.74,
            extinction=0.11,
        ),
        Filter(
            name="i",
            atmospheric_psf_fwhm=0.56,
            sky_brightness=19.7,
            exp_time=1200,
            zeropoint=69.80,
            extinction=0.07,
        ),
        Filter(
            name="y",
            atmospheric_psf_fwhm=0.64,
            sky_brightness=18.3,
            exp_time=1200,
            zeropoint=29.56,
            extinction=0.05,
        ),
        Filter(
            name="z",
            atmospheric_psf_fwhm=0.64,
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
            atmospheric_psf_fwhm=0.703,
            sky_brightness=18.6,
            exp_time=4800,
            zeropoint=10.58,
            extinction=0.138,
        ),
        Filter(
            name="z",
            atmospheric_psf_fwhm=0.725,
            sky_brightness=19.6,
            exp_time=4800,
            zeropoint=22.68,
            extinction=0.043,
        ),
        Filter(
            name="i",
            atmospheric_psf_fwhm=0.748,
            sky_brightness=20.5,
            exp_time=5520,
            zeropoint=32.36,
            extinction=0.07,
        ),
        Filter(
            name="r",
            atmospheric_psf_fwhm=0.781,
            sky_brightness=21.2,
            exp_time=520,
            zeropoint=43.70,
            extinction=0.10,
        ),
        Filter(
            name="g",
            atmospheric_psf_fwhm=0.814,
            sky_brightness=22.3,
            exp_time=2400,
            zeropoint=50.70,
            extinction=0.163,
        ),
        Filter(
            name="u",
            atmospheric_psf_fwhm=0.859,
            sky_brightness=2.9,
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
    mirror_diameter=3.934,
    effective_area=10.014,
    airmass=1.0,
    zeropoint_airmass=1.3,
    filters=[
        Filter(
            name="i",
            atmospheric_psf_fwhm=0.96,
            sky_brightness=20.5,
            exp_time=1000,
            zeropoint=13.94,
            extinction=0.05,
        ),
        Filter(
            name="r",
            atmospheric_psf_fwhm=1.03,
            sky_brightness=21.4,
            exp_time=800,
            zeropoint=15.65,
            extinction=0.09,
        ),
        Filter(
            name="g",
            atmospheric_psf_fwhm=1.24,
            sky_brightness=22.3,
            exp_time=800,
            zeropoint=12.29,
            extinction=0.17,
        ),
        Filter(
            name="z",
            atmospheric_psf_fwhm=1.12,
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
    mirror_diameter=3.592,
    effective_area=8.022,
    airmass=1.0,  # TODO: double-check
    zeropoint_airmass=1.0,  # TODO: double-check
    filters=[
        Filter(
            name="i",
            atmospheric_psf_fwhm=0.64,
            sky_brightness=20.3,
            exp_time=4300,
            zeropoint=8.46,
            extinction=0.07,
        ),
        Filter(
            name="r",
            atmospheric_psf_fwhm=0.71,
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


def get_psf(survey, filt, atmospheric_model="Kolmogorov"):
    """Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)"""

    # get atmospheric psf
    if atmospheric_model == "Kolmogorov":
        atmospheric_psf_model = galsim.Kolmogorov(fwhm=filt.atmospheric_psf_fwhm)
    elif atmospheric_model == "Moffat":
        atmospheric_psf_model = galsim.Moffat(2, fwhm=filt.atmospheric_psf_fwhm)
    else:
        raise NotImplementedError(
            f"The atmospheric model request '{atmospheric_model}' is incorrect or not implemented."
        )

    # get optical psf if available
    if survey.mirror_diameter > 0:
        mirror_area = np.pi * (0.5 * survey.mirror_diameter) ** 2
        area_ratio = survey.effective_area / mirror_area
        if area_ratio <= 0 or area_ratio > 1:
            raise RuntimeError(
                "Incompatible effective-area and mirror-diameter values."
            )
        obscuration_fraction = np.sqrt(1 - area_ratio)
        lambda_over_diameter = 3600 * np.degrees(
            1e-10 * _central_wavelength[filt.name] / survey.mirror_diameter
        )
        optical_psf_model = galsim.Airy(
            lam_over_diam=lambda_over_diameter, obscuration=obscuration_fraction
        )
        psf_model = galsim.Convolve(atmospheric_psf_model, optical_psf_model).withFlux(
            1.0
        )

    else:
        psf_model = atmospheric_model.withFlux(1.0)

    return psf_model
