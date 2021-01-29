import astropy.wcs as WCS
import numpy as np
import galsim
from collections import namedtuple

# A simple class archetype to serve as a dictionary without having to write the field names
# every time.
Survey = namedtuple(
    "Survey",
    [
        "name",
        "pixel_scale",
        "effective_area",
        "mirror_diameter",
        "airmass",
        "filters",
    ],
)

Filter = namedtuple(
    "Filter",
    [
        "name",
        "psf_scale",
        "sky_brightness",
        "exp_time",
        "zenith_psf_fwhm",
        "zero_point",
        "extinction",
    ],
)


pix_roman = 0.11
pix_rubin = 0.2
pix_hst = 0.06
pix_euclid = 0.101
pix_hsc = 0.167
pix_des = 0.263
pix_cfht = 0.185

# Central wavelengths in Angstroms for each LSST filter band, calculated from the
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
}


# https://sci.esa.int/documents/33859/36320/1567253682555-Euclid_presentation_Paris_1Dec2009.pdf
Euclid = Survey("Euclid", pix_euclid, [Filter("VIS", 0.16, 22.9, 2260, 6.85)])

# Source https://hst-docs.stsci.edu/display/WFC3IHB/6.6+UVIS+Optical+Performance#id-6
# .6UVISOpticalPerformance-6.6.1 800nm
HST = Survey("HST", pix_hst, [Filter("f814w", 0.074, 22, 3000, 20)])

# https://hsc-release.mtk.nao.ac.jp/doc/ deep+udeep

HSC = Survey(
    "HSC",
    pix_hsc,
    [
        Filter("g", 0.306, 21.4, 600, 91.11),
        Filter("r", 0.285, 20.6, 600, 87.74),
        Filter("i", 0.238, 19.7, 1200, 69.80),
        Filter("z", 0.268, 18.3, 1200, 29.56),
        Filter("y", 0.272, 17.9, 1200, 21.53),
    ],
)

# Sigma of the psf profile in arcseconds.
# https://arxiv.org/pdf/1702.01747.pdf Z-band
Roman = Survey(
    "Roman",
    pix_roman,
    [
        Filter("F062", 0.1848, 22, 3000, 26.99),
        Filter("Z087", 0.1859, 22, 3000, 26.39),
        Filter("Y106", 0.2046, 22, 3000, 26.41),
        Filter("J129", 0.2332, 22, 3000, 26.35),
        Filter("H158", 0.2684, 22, 3000, 26.41),
        Filter("F184", 0.2981, 22, 3000, 25.96),
    ],  # Mean sky level and exposure time need to be checked
)

# https://www.lsst.org/about/camera/features
Rubin = Survey(
    "LSST",
    pix_rubin,
    32.4,
    8.36,
    1.2,
    [
        Filter("y", 0.327, 18.6, 4800, 0.63, 10.58, 0.138),
        Filter("z", 0.310, 19.6, 4800, 0.65, 22.68, 0.043),
        Filter("i", 0.297, 20.5, 5520, 0.67, 32.36, 0.07),
        Filter("r", 0.285, 21.2, 5520, 0.70, 43.70, 0.10),
        Filter("g", 0.276, 22.3, 2400, 0.73, 50.70, 0.163),
        Filter("u", 0.267, 22.9, 1680, 0.77, 9.16, 0.451),
    ],
)
DES = Survey(
    "DES",
    pix_des,
    [
        Filter("i", None, None, None, None),
        Filter("r", None, None, None, None),
        Filter("g", None, None, None, None),
        Filter("z", None, None, None, None),
    ],  # Will not work properly until we fill in the correct values
)
CFHT = Survey(
    "CFHT",
    pix_cfht,
    [
        Filter("i", None, None, None, None),
        Filter("r", None, None, None, None),
    ],  # Will not work properly until we fill in the correct values
)


def get_filter(survey, band):
    bands = [filt.name for filt in survey.filters]
    assert band in bands
    return survey.filters[bands.index(band)]


def get_flux(ab_magnitude, filt, survey):
    """Convert source magnitude to flux.
    The calculation includes the effects of atmospheric extinction.
    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        ab_magnitude(float): AB magnitude of source.
    Returns:
        float: Flux in detected electrons.
    """
    zeropoint_airmass = 1.0
    if survey.name == "DES":
        zeropoint_airmass = 1.3
    if survey.name == "LSST" or survey.name == "HSC":
        zeropoint_airmass = 1.2
    if survey.name == "Euclid":
        zeropoint_airmass = 1.0

    mag = ab_magnitude + filt.extinction * (filt.airmass - zeropoint_airmass)
    return filt.exposure_time * filt.zero_point * 10 ** (-0.4 * (mag - 24))


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


def get_moffat_psf(survey, filt, psf_stamp_size):
    """Generates a psf as a Galsim object using the survey information"""
    assert psf_stamp_size % 2 == 1

    def psf_function(r):
        return galsim.Moffat(2, r)

    psf_obj = psf_function(filt.psf_scale).withFlux(1.0)
    psf = psf_obj.drawImage(
        nx=psf_stamp_size,
        ny=psf_stamp_size,
        method="no_pixel",
        use_true_center=True,
        scale=survey.pixel_scale,
    ).array

    # Make sure PSF vanishes on the edges of a patch that
    # has the shape of the initial psf
    psf = psf - psf[1, int(psf_stamp_size / 2)] * 2
    psf[psf < 0] = 0
    psf /= np.sum(psf)

    # Generating an unintegrated galsim psf for the convolution
    psf_obj = galsim.InterpolatedImage(
        galsim.Image(psf), scale=survey.pixel_scale
    ).withFlux(1.0)

    return psf_obj


def get_psf(survey):
    """Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)"""
    psfs = []  # one per filter.
    for filt in survey.filters:

        # get optical psf
        assert survey.mirror_diameter > 0
        area_ratio = survey.effective_area / (np.pi * (0.5 * survey.mirror_diameter) ** 2)
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

        # get atmospheric psf
        atmospheric_psf_fwhm = filt.zenith_psf_fwhm * filt.airmass ** 0.6
        atmospheric_psf_model = galsim.Kolmogorov(fwhm=atmospheric_psf_fwhm)

        # combine them and obtain psf image.
        psf_model = galsim.Convolve(atmospheric_psf_model, optical_psf_model)
        psfs.append(psf_model)
    return psfs
