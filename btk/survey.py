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
        "fwhm", # in arcsec: must be None is psf_path is used
        "psf_path", # path to retrieve the PSF from a FITS image: must be None if fwhm is used
        "sky_brightness",  # mags/sq.arcsec
        "exp_time",  # in seconds [s]
        "zeropoint",  # in electrons per second at 24th magnitude.
        "extinction",  # Exponential extinction coefficient for atmospheric absorption.
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


def define_synthetic_psf(mirror_diameter, effective_area, filt_wavelength, fwhm, atmospheric_model="Kolmogorov"):
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
        psf_model = galsim.Convolve(atmospheric_psf_model, optical_psf_model).withFlux(
            1.0
        )

    else:
        psf_model = atmospheric_model.withFlux(1.0)
        
    return psf_model


def make_psf(survey, filt):
    """Generate a PSF galsim model
    Args:
        survey (Survey): survey to use
        filt (Filter): filter to use
    Returns:
        psf_model: galsim PSF model
    """

    assert((filt.fwhm==None) ^ (filt.psf_path==None))

    # synthetic PSF model
    if filt.psf_path==None:
        if type(filt.fwhm)==list:
            fwhm = rd.choice(filt.fwhm)
        else:
            fwhm = filt.fwhm
        psf_model = define_synthetic_psf(survey.mirror_diameter, survey.effective_area, _central_wavelength[filt.name], fwhm)
    # FITS image PSF model
    else:
        if type(filt.psf_path)==list:
            psf_file = rd.choice(filt.psf_path)
        else:
            psf_file = filt.psf_path
        psf_array = fits.getdata(psf_file)
        psf_model = galsim.InterpolatedImage(galsim.Image(psf_array), scale=survey.pixel_scale).withFlux(1.0)
        
    return psf_model

        
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
            psf=define_synthetic_psf(1.3, 1.15, _central_wavelength["VIS"], 0.17),
            fwhm=0.17,
            psf_path=None,
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
            psf=define_synthetic_psf(2.4, 1.0, _central_wavelength["f814w"], 0.074),
            fwhm=0.074,
            psf_path=None,
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
            psf=define_synthetic_psf(8.2, 52.81, _central_wavelength["g"], 0.72),
            fwhm=0.72,
            psf_path=None,
            sky_brightness=21.4,
            exp_time=600,
            zeropoint=91.11,
            extinction=0.13,
        ),
        Filter(
            name="r",
            psf=define_synthetic_psf(8.2, 52.81, _central_wavelength["r"], 0.67),
            fwhm=0.67,
            psf_path=None,
            sky_brightness=20.6,
            exp_time=600,
            zeropoint=87.74,
            extinction=0.11,
        ),
        Filter(
            name="i",
            psf=define_synthetic_psf(8.2, 52.81, _central_wavelength["i"], 0.56),
            fwhm=0.56,
            psf_path=None,
            sky_brightness=19.7,
            exp_time=1200,
            zeropoint=69.80,
            extinction=0.07,
        ),
        Filter(
            name="y",
            psf=define_synthetic_psf(8.2, 52.81, _central_wavelength["y"], 0.64),
            fwhm=0.64,
            psf_path=None,
            sky_brightness=18.3,
            exp_time=1200,
            zeropoint=29.56,
            extinction=0.05,
        ),
        Filter(
            name="z",
            psf=define_synthetic_psf(8.2, 52.81, _central_wavelength["z"], 0.64),
            fwhm=0.64,
            psf_path=None,
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
            psf=make_psf,
            fwhm=0.703,
            psf_path=None,
            sky_brightness=18.6,
            exp_time=4800,
            zeropoint=10.58,
            extinction=0.138,
        ),
        Filter(
            name="z",
            psf=make_psf,
            fwhm=0.725,
            psf_path=None,
            sky_brightness=19.6,
            exp_time=4800,
            zeropoint=22.68,
            extinction=0.043,
        ),
        Filter(
            name="i",
            psf=make_psf,
            fwhm=0.748,
            psf_path=None,
            sky_brightness=20.5,
            exp_time=5520,
            zeropoint=32.36,
            extinction=0.07,
        ),
        Filter(
            name="r",
            psf=make_psf,
            fwhm=0.781,
            psf_path=None,
            sky_brightness=21.2,
            exp_time=5520,
            zeropoint=43.70,
            extinction=0.10,
        ),
        Filter(
            name="g",
            psf=make_psf,
            fwhm=0.814,
            psf_path=None,
            sky_brightness=22.3,
            exp_time=2400,
            zeropoint=50.70,
            extinction=0.163,
        ),
        Filter(
            name="u",
            psf=make_psf,
            fwhm=0.859,
            psf_path=None,
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
            psf=define_synthetic_psf(3.934, 10.014, _central_wavelength["i"], 0.96),
            fwhm=0.96,
            psf_path=None,
            sky_brightness=20.5,
            exp_time=1000,
            zeropoint=13.94,
            extinction=0.05,
        ),
        Filter(
            name="r",
            psf=define_synthetic_psf(3.934, 10.014, _central_wavelength["r"], 1.03),
            fwhm=1.03,
            psf_path=None,
            sky_brightness=21.4,
            exp_time=800,
            zeropoint=15.65,
            extinction=0.09,
        ),
        Filter(
            name="g",
            psf=define_synthetic_psf(3.934, 10.014, _central_wavelength["g"], 1.24),
            fwhm=1.24,
            psf_path=None,
            sky_brightness=22.3,
            exp_time=800,
            zeropoint=12.29,
            extinction=0.17,
        ),
        Filter(
            name="z",
            psf=define_synthetic_psf(3.934, 10.014, _central_wavelength["z"], 1.12),
            fwhm=1.12,
            psf_path=None,
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
            psf=define_synthetic_psf(3.592, 8.022, _central_wavelength["i"], 0.64),
            fwhm=0.64,
            psf_path=None,
            sky_brightness=20.3,
            exp_time=4300,
            zeropoint=8.46,
            extinction=0.07,
        ),
        Filter(
            name="r",
            psf=define_synthetic_psf(3.592, 8.022, _central_wavelength["r"], 0.71),
            fwhm=0.71,
            psf_path=None,
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
