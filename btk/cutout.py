import descwl
from abc import ABC
import astropy.wcs as WCS


def make_wcs(
    pixel_scale, shape, center_pix=None, center_sky=None, projection=None
):
    """Creates WCS for an image.
    Args:
        pixel_scale (float): pixel size in arcseconds
        shape (tuple): shape of the image in pixels.
        center_pix (tuple): position of the reference pixel used as the center of the
                            affine transform for the wcs.
        center_sky (list): sky coordinates corresponding to center_pix, in arcseconds
        projection(str): projection type, default to TAN. A list of available
                            types can be found in astropy.wcs documentation
    Returns:
        wcs: WCS
    """
    if center_pix is None:
        center_pix = [(s + 1) / 2 for s in shape]
    if center_sky is None:
        center_sky = [0 for _ in range(naxis)]
    if projection is None:
        projection = "TAN"
    w = WCS.WCS(naxis=2)
    w.wcs.ctype = ["RA---" + projection, "DEC--" + projection]
    w.wcs.crpix = center_pix
    w.wcs.cdelt = [pixel_scale / 3600 for _ in range(naxis)]
    w.wcs.crval = [c / 3600 for c in center_sky]
    w.array_shape = shape
    return w


class Cutout(ABC):
    def __init__(
        self, stamp_size, pixel_scale, center_pix=None, center_sky=None, projection=None
    ):
        """Class containing the necessary information to draw a postage stamp (PSF,
        pixel_scale, WCS, etc.) for a given survey and band.
        """
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.pix_stamp_size = int(self.stamp_size / pixel_scale)
        self.wcs = self.get_wcs(center_pix, center_sky, projection)

    def get_wcs(self, center_pix=None, center_sky=None, projection=None):
        return make_wcs(
            pixel_scale=self.pixel_scale,
            center_pix=center_pix,
            center_sky=center_sky,
            projection=projection,
            shape=(self.pix_stamp_size, self.pix_stamp_size),
        )


class WLDCutout(descwl.survey.Survey, Cutout):
    """Extension of the descwl survey class including information for the WCS.
    Args:
        center_pix: tuple representing the center of the image in pixels
        center_sky: tuple representing the center of the image in sky coordinates
                     (RA,DEC) in arcseconds.
        projection: string representing the type of projection for the WCS. If None,
                     it will default to "TAN". A list of available projections can
                     be found in the documentation of `astropy.wcs`
        wcs: an `astropy.wcs.wcs` object corresponding to the parameters center_pix,
              center_sky, projection, pixel_scale and stamp_size.
        **survey_kwargs: any arguments given to a descwl survey
    """

    def __init__(
        self,
        stamp_size,
        center_pix=None,
        center_sky=None,
        projection=None,
        **survey_kwargs,
    ):
        descwl.survey.Survey.__init__(self, **survey_kwargs)
        Cutout.__init__(
            self,
            stamp_size,
            survey_kwargs["pixel_scale"],
            center_pix,
            center_sky,
            projection,
        )

    def get_psf_sky(self, psf_stamp_size):
        """Returns postage stamp image of the PSF and mean background sky
        level value saved in the input obs_conds class
        Args:
            psf_stamp_size: Size of postage stamp to draw PSF on in pixels.
        Returns:
            psf_image (np.ndarray): Postage stamp image of PSF
            mean_sky_level (float): Mean of sky background
        """
        mean_sky_level = self.mean_sky_level
        psf = self.psf_model
        psf_image = psf.drawImage(
            scale=self.pixel_scale, nx=psf_stamp_size, ny=psf_stamp_size
        ).array
        return psf_image, mean_sky_level
