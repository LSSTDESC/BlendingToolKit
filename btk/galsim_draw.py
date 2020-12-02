import galsim
from astropy import wcs as WCS
import numpy as np
import scarlet

center_ra = 19.3 * galsim.degrees  # The RA, Dec of the center of the image on the sky
center_dec = -33.1 * galsim.degrees


def mk_sim(k, dir, shape, npsf, cat, shift=(0, 0), gal_type="real"):
    """creates low and high resolution images of a galaxy profile with different psfs from the list of galaxies in the COSMOS catalog

    Parameters
    ----------
    k: int
        index of the galaxy to draw from the COSMOS catalog
    dir: dictionary
        dictionary that contains the information for the high resolution survey
    shape_lr: tuple of ints
        shape of the lr image
    npsf: int
        size on-a-side of a psf (in pixels)
    cat: list
        catalog where to draw galaxies from

    Returns
    -------
    im_hr: galsim Image
        galsim Image object with the high resolution simulated image and its WCS
    im_lr: galsim Image
        galsim Image object with the low resolution simulated image and its WCS
    psf_hr: numpy array
        psf of the high resolution image
    psf_lr: numpy array
        psf of the low resolution image
    """
    pix = dir["pixel"]
    sigma = dir["psf"]

    # Rotation angle
    theta = np.random.randn(1) * np.pi * 0
    angle = galsim.Angle(theta, galsim.radians)

    # Image frames
    im = galsim.Image(shape[0], shape[1], scale=pix)

    # Galaxy profile
    gal = cat.makeGalaxy(k, gal_type=gal_type, noise_pad_size=shape[0] * pix)
    gal = gal.shift(dx=shift[0], dy=shift[1])

    ## PSF is a Moffat profile dilated to the sigma of the corresponding survey
    psf_int = galsim.Moffat(2, HST["pixel"]).dilate(sigma / HST["psf"]).withFlux(1.0)
    ## Draw PSF
    psf = psf_int.drawImage(
        nx=npsf, ny=npsf, method="real_space", use_true_center=True, scale=pix_hr
    ).array
    ## Make sure PSF vanishes on the edges of a patch that has the shape of the initial npsf
    psf = psf - psf[0, int(npsf / 2)] * 2
    psf[psf < 0] = 0
    psf = psf / np.sum(psf)
    ## Interpolate the new 0-ed psf
    psf_int = galsim.InterpolatedImage(galsim.Image(psf), scale=pix).withFlux(1.0)
    ## Re-draw it (with the correct fulx)
    psf = psf_int.drawImage(
        nx=npsf, ny=npsf, method="real_space", use_true_center=True, scale=pix_hr
    ).array
    # Convolve galaxy profile by PSF, rotate and sample at high resolution
    im = galsim.Convolve(gal, psf_int).drawImage(
        nx=shape[0],
        ny=shape[1],
        use_true_center=True,
        method="no_pixel",
        scale=pix,
        dtype=np.float64,
    )
    return im


def mk_scene(hr_dict, lr_dict, cat, shape_hr, shape_lr, n_gal, gal_type):
    """Generates blended scenes at two resolutions

    Parameters
    ----------
    hr_dict, lr_dict: dictionaries
        two dictionaries that contain the information for surveys (pixel scale, name and psf size)
    cat: catalog
        catalog of sources for galsim
    shape_hr, shape_lr: tuples
        2D shapes of the desired images for surveys indicated in the dictionaries
    ngal: int
        number of galaxies to draw on the scene
    """
    pix_hr = hr_dict["pixel"]

    lr = 0
    hr = 0
    loc = []
    for i in range(n_gal):
        k = np.int(np.random.rand(1) * len(cat))
        shift = (np.random.rand(2) - 0.5) * shape_hr * pix_hr / 2
        ihr, ilr, phr, plr, _ = mk_sim(
            k,
            hr_dict,
            lr_dict,
            shape_hr,
            shape_lr,
            41,
            cat,
            shift=shift,
            gal_type=gal_type,
        )
        sed_lr = np.random.rand(3) * 0.8 + 0.2
        sed_hr = np.random.rand(1) * 15
        hr += ihr.array * sed_hr
        lr += ilr.array[None, :, :] * sed_lr[:, None, None]
        loc.append(
            [shift[0] / pix_hr + shape_hr[0] / 2, shift[1] / pix_hr + shape_hr[1] / 2]
        )
    lr += (
        np.random.randn(*lr.shape)
        * np.sum(lr ** 2) ** 0.5
        / np.size(lr)
        * 10
        / sed_lr[:, None, None]
    )
    hr += np.random.randn(*hr.shape) * np.max(hr) / 200
    plr = plr * np.ones(3)[:, None, None]
    return hr, lr, ihr.wcs, ilr.wcs, phr, plr, np.array(loc)


def setup_scarlet(data_hr, data_lr, psf_hr, psf_lr, channels, coverage="union"):
    """Performs the initialisation steps for scarlet to run its resampling scheme

    Prameters
    ---------
    data_hr: galsim Image
        galsim Image object with the high resolution simulated image and its WCS
    data_lr: galsim Image
        galsim Image object with the low resolution simulated image and its WCS
    psf_hr: numpy array
        psf of the high resolution image
    psf_lr: numpy array
        psf of the low resolution image
    channels: tuple
        names of the channels

    Returns
    -------
    obs: array of observations
        array of scarlet.Observation objects initialised for resampling
    """
    # Extract data
    im_hr = data_hr.array[None, :, :]
    im_lr = data_lr.array[None, :, :]

    # define two observation objects and match to frame
    obs_hr = scarlet.Observation(
        im_hr, wcs=data_hr.wcs, psfs=psf_hr, channels=[channels[1]]
    )
    obs_lr = scarlet.Observation(
        im_lr, wcs=data_lr.wcs, psfs=psf_lr, channels=[channels[0]]
    )

    # Keep the order of the observations consistent with the `channels` parameter
    # This implementation is a bit of a hack and will be refined in the future
    obs = [obs_lr, obs_hr]

    scarlet.Frame.from_observations(obs, obs_id=1, coverage=coverage)
    return obs


def interp_galsim(data_hr, data_lr, diff_psf, angle, h_hr, h_lr):
    """Apply resampling from galsim

    Prameters
    ---------
    data_hr: galsim Image
        galsim Image object with the high resolution simulated image and its WCS
    data_lr: galsim Image
        galsim Image object with the low resolution simulated image and its WCS
    diff_hr: numpy array
        difference kernel betwee the high and low resolution psf
    angle: float
        angle between high and low resolution images
    h_hr: float
        scale of the high resolution pixel (arcsec)
    h_lr: float
        scale of the low resolution pixel (arcsec)

    Returns
    -------
    interp_gal: galsim.Image
        image interpolated at low resolution
    """
    # Load data
    im_hr = data_hr.array[None, :, :]
    im_lr = data_lr.array[None, :, :]
    _, n_hr, n_hr = im_hr.shape
    _, n_lr, n_lr = im_lr.shape

    # Interpolate hr image
    gal_hr = galsim.InterpolatedImage(galsim.Image(im_hr[0]), scale=h_hr)

    # Rotate hr galaxy to lr frame
    rot_gal = gal_hr.rotate(galsim.Angle(angle, galsim.radians))

    # Convolve hr galaxy by diff kernel at hr
    conv_gal = galsim.Convolve(rot_gal, diff_psf)

    # Downsamples to low resolution
    interp_gal = conv_gal.drawImage(
        nx=n_lr,
        ny=n_lr,
        scale=h_lr,
        method="no_pixel",
    )

    return interp_gal


def SDR(X_true, X):
    """Source distortion ratio between an expected value and its estimate. The higher the SDR the better X_true and X agree"""
    return 10 * np.log10(np.sum(X_true ** 2) ** 0.5 / np.sum((X_true - X) ** 2) ** 0.5)
