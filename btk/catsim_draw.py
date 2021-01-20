"""Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)"""
import math
import numpy as np
import galsim

surveys = {
    "LSST": {
        "i": {
            "mirror_diameter": 8.36,
            "effective_area": 32.4,
            "pixel_scale": 0.2,
            "exposure_time": 5520.0,
            "sky_brightness": 20.5,
            "zenith_psf_fwhm": 0.67,
            "zeropoint": 32.36,
            "extinction": 0.07,
            "airmass": 1.2,
            "central_wavelength": 7528.51,
        }
    }
}


class SourceNotVisible(Exception):
    """Custom exception to indicate that a source has no visible model components."""

    pass


def get_filter(survey, band):
    bands = [filt.name for filt in survey.filters]
    assert band in bands
    return survey.filters[bands.index(band)]


def get_flux(ab_magnitude, survey, band="i"):
    """Convert source magnitude to flux.
    The calculation includes the effects of atmospheric extinction.
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

    filt = get_filter(survey, band)
    mag = ab_magnitude + filt.extinction * (filt.airmass - zeropoint_airmass)
    return filt.exposure_time * filt.zero_point * 10 ** (-0.4 * (mag - 24))


def sersic_second_moments(n, hlr, q, beta):
    """Calculate the second-moment tensor of a sheared Sersic radial profile.
    Args:
        n(int): Sersic index of radial profile. Only n = 1 and n = 4 are supported.
                * n=1 corresponds to Exponential/Disk profile
                * n=4 corresponds to de Vaucouleurs/Bulge profile
        hlr(float): Radius of 50% isophote before shearing, in arcseconds.
        q(float): Ratio b/a of Sersic isophotes after shearing.
        beta(float): Position angle of sheared isophotes in radians, measured anti-clockwise
            from the positive x-axis.
    Returns:
        numpy.ndarray: Array of shape (2,2) with values of the second-moments tensor
            matrix, in units of square arcseconds.
    Raises:
        RuntimeError: Invalid Sersic index n.
    """
    # Lookup the value of cn = 0.5*(r0/hlr)**2 Gamma(4*n)/Gamma(2*n)
    if n == 1:
        cn = 1.06502
    elif n == 4:
        cn = 10.8396
    else:
        raise RuntimeError("Invalid Sersic index n.")
    e_mag = (1.0 - q) / (1.0 + q)
    e_mag_sq = e_mag ** 2
    e1 = e_mag * math.cos(2 * beta)
    e2 = e_mag * math.sin(2 * beta)
    Q11 = 1 + e_mag_sq + 2 * e1
    Q22 = 1 + e_mag_sq - 2 * e1
    Q12 = 2 * e2
    return np.array(((Q11, Q12), (Q12, Q22))) * cn * hlr ** 2 / (1 - e_mag_sq) ** 2


def moments_size_and_shape(Q: np.ndarray):
    """Calculate size and shape parameters from a second-moment tensor.
    If the input is an array of second-moment tensors, the calculation is vectorized
    and returns a tuple of output arrays with the same leading dimensions (...).
    Args:
        Q(numpy.ndarray): Array of shape (...,2,2) containing second-moment tensors,
            which are assumed to be symmetric (only the [0,1] component is used).
    Returns:
        tuple: Tuple (sigma_m,sigma_p,a,b,beta,e1,e2) of :class:`numpy.ndarray` objects
            with shape (...). Refer to :ref:`analysis-results` for details on how
            each of these vectors is defined.
    """
    trQ = np.trace(Q, axis1=-2, axis2=-1)
    detQ = np.linalg.det(Q)
    sigma_m = np.power(detQ, 0.25)
    sigma_p = np.sqrt(0.5 * trQ)
    asymQx = Q[..., 0, 0] - Q[..., 1, 1]
    asymQy = 2 * Q[..., 0, 1]
    asymQ = np.sqrt(asymQx ** 2 + asymQy ** 2)
    a = np.sqrt(0.5 * (trQ + asymQ))
    b = np.sqrt(0.5 * (trQ - asymQ))
    beta = 0.5 * np.arctan2(asymQy, asymQx)
    e_denom = trQ + 2 * np.sqrt(detQ)
    e1 = asymQx / e_denom
    e2 = asymQy / e_denom
    return sigma_m, sigma_p, a, b, beta, e1, e2


def get_ellipticity(a, b, pa):
    # a, b are semi-major/minor axis
    # beta in radians is assumed.
    beta = math.radians(pa)
    q = b / a
    e_mag = (1.0 - q) / (1.0 + q)
    e1 = e_mag * math.cos(2 * beta)
    e2 = e_mag * math.sin(2 * beta)
    return e1, e2, beta


def get_hlr(a, b):
    hlr_arcsecs = math.sqrt(a * b)
    return hlr_arcsecs


def get_galaxy(entry, survey, band, no_disk=False, no_bulge=False, no_agn=False):
    components = []

    total_flux = get_flux(entry["ab_magnitude"], survey, band)
    # Calculate the flux of each component in detected electrons.
    total_fluxnorm = (
        entry["fluxnorm_disk"] + entry["fluxnorm_bulge"] + entry["fluxnorm_agn"]
    )
    disk_flux = 0.0 if no_disk else entry["fluxnorm_disk"] / total_fluxnorm * total_flux
    bulge_flux = (
        0.0 if no_bulge else entry["fluxnorm_bulge"] / total_fluxnorm * total_flux
    )
    agn_flux = 0.0 if no_agn else entry["fluxnorm_agn"] / total_fluxnorm * total_flux

    if disk_flux + bulge_flux + agn_flux == 0:
        raise SourceNotVisible

    # Calculate the position of angle of the Sersic components, which are assumed to be the same.
    if disk_flux > 0:
        beta_radians = math.radians(entry["pa_disk"])
        if bulge_flux > 0:
            assert (
                entry["pa_disk"] == entry["pa_bulge"]
            ), "Sersic components have different beta."
    elif bulge_flux > 0:
        beta_radians = math.radians(entry["pa_bulge"])
    else:
        # This might happen if we only have an AGN component.
        beta_radians = None
    # Calculate shapes hlr = sqrt(a*b) and q = b/a of Sersic components.
    if disk_flux > 0:
        a_d, b_d = entry["a_d"], entry["b_d"]
        disk_hlr_arcsecs = math.sqrt(a_d * b_d)
        disk_q = b_d / a_d
    else:
        disk_hlr_arcsecs, disk_q = None, None
    if bulge_flux > 0:
        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = math.sqrt(a_b * b_b)
        bulge_q = b_b / a_b
    else:
        bulge_hlr_arcsecs, bulge_q = None, None

    if disk_flux > 0:
        disk = galsim.Exponential(
            flux=disk_flux, half_light_radius=disk_hlr_arcsecs
        ).shear(q=disk_q, beta=beta_radians * galsim.radians)
        components.append(disk)

    a_b, b_b = entry["a_b"], entry["b_b"]
    bulge_hlr_arcsecs = math.sqrt(a_b * b_b)
    bulge_q = b_b / a_b

    if disk_flux > 0:
        disk = galsim.Exponential(
            flux=disk_flux, half_light_radius=disk_hlr_arcsecs
        ).shear(q=disk_q, beta=beta_radians * galsim.radians)
        components.append(disk)

    if bulge_flux > 0:
        bulge = galsim.DeVaucouleurs(
            flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs
        ).shear(q=bulge_q, beta=beta_radians * galsim.radians)
        components.append(bulge)

    if agn_flux > 0:
        agn = galsim.Gaussian(flux=agn_flux, sigma=1e-8)
        components.append(agn)

    profile = galsim.Add(components)

    return profile


def get_psf(survey, band):
    filt = get_filter(survey, band)

    atmospheric_psf_fwhm = filt.zenith_psf_fwhm * filt.airmass ** 0.6
    area_ratio = filt.effective_area / (math.pi * (0.5 * filt.mirror_diameter) ** 2)
    obscuration_fraction = math.sqrt(1 - area_ratio)
    lambda_over_diameter = 3600 * math.degrees(
        1e-10 * filt.central_wavelength / filt.mirror_diameter
    )

    atmospheric_psf_model = galsim.Kolmogorov(fwhm=atmospheric_psf_fwhm)
    optical_psf_model = galsim.Airy(
        lam_over_diam=lambda_over_diameter, obscuration=obscuration_fraction
    )
    psf_model = galsim.Convolve(atmospheric_psf_model, optical_psf_model)

    psf_size_pixels = 2 * int(math.ceil(10 * atmospheric_psf_fwhm / survey.pixel_scale))
    psf_image = galsim.Image(psf_size_pixels, psf_size_pixels, scale=survey.pixel_scale)
    return psf_model, psf_image
