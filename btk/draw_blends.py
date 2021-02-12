import copy
from itertools import chain
from abc import ABC, abstractmethod

import galsim
import numpy as np
from astropy.table import Column


from btk.multiprocess import multiprocess
from btk.create_blend_generator import BlendGenerator
from btk.survey import get_mean_sky_level, get_psf, make_wcs, get_flux


class SourceNotVisible(Exception):
    """Custom exception to indicate that a source has no visible model components."""


def get_center_in_pixels(blend_catalog, wcs):
    """Returns center of objects in blend_catalog in pixel coordinates of
    postage stamp.

    blend_catalog contains ra dec of object center with the postage stamp
    center being 0,0. The size of the postage stamp and pixel scale is used to
    compute the object centers in pixel coordinates. Coordinates are in pixels
    where bottom left corner of postage stamp is (0, 0).

    Args:
        blend_catalog: Catalog with entries corresponding to one blend.
        wcs: astropy.wcs.WCS object corresponding to the image
    Returns:
        `astropy.table.Column`: x and y coordinates of object centroid
    """
    dx, dy = wcs.all_world2pix(
        blend_catalog["ra"] / 3600, blend_catalog["dec"] / 3600, 0
    )
    dx_col = Column(dx, name="dx")
    dy_col = Column(dy, name="dy")
    return dx_col, dy_col


def get_size(catalog, psf, pixel_scale):
    """Returns a astropy.table.column with the size of the galaxy in pixels.

    Galaxy size is estimated as second moments size (r_sec) computed as
    described in A1 of Chang et.al 2012. The PSF second moment size, psf_r_sec,
    is computed by galsim from the PSF model in obs_conds in the i band.
    The object size is the defined as sqrt(r_sec**2 + 2*psf_r_sec**2).

    Args:
        catalog: Catalog with entries corresponding to one blend.
        psf: Galsim object corresponding to a PSF.
        pixel_scale: arcseconds per pixel

    Returns:
        `astropy.table.Column`: size of the galaxy in pixels.
    """

    r_sec = catalog["btk_size"]
    psf_r_sec = psf.calculateMomentRadius()
    size = np.sqrt(r_sec ** 2 + psf_r_sec ** 2) / pixel_scale
    return Column(size, name="size")


def get_catsim_galaxy(entry, filt, survey, no_disk=False, no_bulge=False, no_agn=False):
    """Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)"""

    components = []
    total_flux = get_flux(entry[filt.name + "_ab"], filt, survey)
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

    if disk_flux > 0:
        beta_radians = np.radians(entry["pa_disk"])
        if bulge_flux > 0:
            assert (
                entry["pa_disk"] == entry["pa_bulge"]
            ), "Sersic components have different beta."
        a_d, b_d = entry["a_d"], entry["b_d"]
        disk_hlr_arcsecs = np.sqrt(a_d * b_d)
        disk_q = b_d / a_d
        disk = galsim.Exponential(
            flux=disk_flux, half_light_radius=disk_hlr_arcsecs
        ).shear(q=disk_q, beta=beta_radians * galsim.radians)
        components.append(disk)

    if bulge_flux > 0:
        beta_radians = np.radians(entry["pa_bulge"])
        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
        bulge_q = b_b / a_b
        bulge = galsim.DeVaucouleurs(
            flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs
        ).shear(q=bulge_q, beta=beta_radians * galsim.radians)
        components.append(bulge)

    if agn_flux > 0:
        agn = galsim.Gaussian(flux=agn_flux, sigma=1e-8)
        components.append(agn)

    profile = galsim.Add(components)
    return profile


class DrawBlendsGenerator(ABC):

    compatible_catalogs = ("Catalog",)

    def __init__(
        self,
        catalog,
        sampling_function,
        surveys: list,
        batch_size=8,
        stamp_size=24,
        meas_bands=("i",),
        multiprocessing=False,
        cpus=1,
        verbose=False,
        add_noise=True,
        shifts=None,
        indexes=None,
        atmospheric_model="Kolmogorov",
        snr=200,
    ):
        """Class that generates images of blended objects, individual isolated
        objects, for each blend in the batch.

        Batch is divided into mini batches of size blend_generator.batch_size//cpus and
        each mini-batch analyzed separately. The results are then combined to output a
        dict with results of entire batch. If multiprocessing is true, then each of
        the mini-batches are run in parallel.

        """

        self.blend_generator = BlendGenerator(
            catalog, sampling_function, batch_size, shifts, indexes, verbose
        )
        self.catalog = self.blend_generator.catalog
        self.multiprocessing = multiprocessing
        self.cpus = cpus

        self.batch_size = self.blend_generator.batch_size
        self.max_number = self.blend_generator.max_number

        if not isinstance(surveys, list):
            raise TypeError("surveys must be a list of Survey objects.")
        self.surveys = surveys
        self.stamp_size = stamp_size
        self.snr = snr

        self.meas_bands = {}
        for i, s in enumerate(self.surveys):
            self.meas_bands[s.name] = meas_bands[i]

        self.add_noise = add_noise
        self.verbose = verbose

        # psf
        self.atmospheric_model = atmospheric_model

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns:
            Dictionary with blend images, isolated object images, blend catalog,
            and observing conditions.
        """
        batch_blend_cat, batch_obs_cond = {}, {}
        blend_images = {}
        isolated_images = {}
        blend_cat = next(self.blend_generator)
        mini_batch_size = np.max([self.batch_size // self.cpus, 1])
        psfs = {}
        wcss = {}

        for s in self.surveys:
            pix_stamp_size = int(self.stamp_size / s.pixel_scale)

            # create WCS and PSF information
            psf = [get_psf(s, filt, self.atmospheric_model) for filt in s.filters]
            wcs = make_wcs(s.pixel_scale, (pix_stamp_size, pix_stamp_size))
            psfs[s.name] = psf
            wcss[s.name] = wcs

            # allocate memory for output catalogues and images.
            batch_blend_cat[s.name] = []
            batch_obs_cond[s.name] = []
            image_shape = (
                pix_stamp_size,
                pix_stamp_size,
                len(s.filters),
            )
            blend_images[s.name] = np.zeros((self.batch_size, *image_shape))
            isolated_images[s.name] = np.zeros(
                (self.batch_size, self.max_number, *image_shape)
            )

            input_args = []
            for i in range(0, self.batch_size, mini_batch_size):
                cat = copy.deepcopy(blend_cat[i : i + mini_batch_size])
                input_args.append((cat, psf, wcs, s))

            # multiprocess and join results
            # ideally, each cpu processes a single mini_batch
            mini_batch_results = multiprocess(
                self.render_mini_batch,
                input_args,
                self.cpus,
                self.multiprocessing,
                self.verbose,
            )

            # join results across mini-batches.
            batch_results = list(chain(*mini_batch_results))

            # organize results.
            for i in range(self.batch_size):
                blend_images[s.name][i] = batch_results[i][0]
                isolated_images[s.name][i] = batch_results[i][1]
                batch_blend_cat[s.name].append(batch_results[i][2])
        if len(self.surveys) > 1:
            output = {
                "blend_images": blend_images,
                "isolated_images": isolated_images,
                "blend_list": batch_blend_cat,
                "psf": psfs,
                "wcs": wcss,
            }
        else:
            survey_name = self.surveys[0].name
            output = {
                "blend_images": blend_images[survey_name],
                "isolated_images": isolated_images[survey_name],
                "blend_list": batch_blend_cat[survey_name],
                "psf": psfs[survey_name],
                "wcs": wcss[survey_name],
            }
        return output

    def render_mini_batch(self, blend_list, psf, wcs, survey):
        """Returns isolated and blended images for blend catalogs in blend_list

        Function loops over blend_list and draws blend and isolated images in each
        band. Even though blend_list was input to the function, we return it since,
        the blend catalogs now include additional columns that flag if an object
        was not drawn and object centers in pixel coordinates.

        Args:
            blend_list (list): List of catalogs with entries corresponding to one
                               blend. The size of this list is equal to the
                               mini_batch_size.
            cutouts (list): List of `btk.cutout.Cutout` objects describing
                            observing conditions in different bands for given survey
                            `survey_name`. The order of cutouts corresponds to order in
                            `survey['bands']`.
            survey (dict): Dictionary containing survey information.

        Returns:
            `numpy.ndarray` of blend images and isolated galaxy images, along with
            list of blend catalogs.
        """
        outputs = []
        for i, blend in enumerate(blend_list):

            # All bands in same survey have same pixel scale, WCS
            pixel_scale = survey.pixel_scale
            pix_stamp_size = int(self.stamp_size / pixel_scale)

            # Band to do measurements of size for given survey.
            meas_band = self.meas_bands[survey.name]
            indx_meas_band = [filt.name for filt in survey.filters].index(meas_band)

            dx, dy = get_center_in_pixels(blend, wcs)
            blend.add_column(dx)
            blend.add_column(dy)
            # TODO: How to get size for COSMOS?
            if "CatsimCatalog" in self.compatible_catalogs:
                size = get_size(blend, psf[indx_meas_band], pixel_scale)
                blend_list[i].add_column(size)

            iso_image_multi = np.zeros(
                (
                    self.max_number,
                    pix_stamp_size,
                    pix_stamp_size,
                    len(survey.filters),
                )
            )
            blend_image_multi = np.zeros(
                (pix_stamp_size, pix_stamp_size, len(survey.filters))
            )
            for b, filter in enumerate(survey.filters):
                single_band_output = self.render_blend(blend, psf[b], filter, survey)
                blend_image_multi[:, :, b] = single_band_output[0]
                iso_image_multi[:, :, :, b] = single_band_output[1]

            outputs.append([blend_image_multi, iso_image_multi, blend])
        return outputs

    def render_blend(self, blend_catalog, psf, filt, survey):
        """Draws image of isolated galaxies along with the blend image in the
        single input band.

        The WLDeblending package (descwl) renders galaxies corresponding to the
        blend_catalog entries and with observing conditions determined by
        cutout. The rendered objects are stored in the observing conditions
        class. So as to not overwrite images across different blends, we make a
        copy of the cutout while drawing each galaxy. Images of isolated
        galaxies are drawn with the WLDeblending and them summed to produce the
        blend image.

        A column 'not_drawn_{band}' is added to blend_catalog initialized as zero.
        If a galaxy was not drawn by descwl, then this flag is set to 1.

        Args:
            blend_catalog: Catalog with entries corresponding to one blend.
            filt(string): Name of filter to draw images in.

        Returns:
            Images of blend and isolated galaxies as `numpy.ndarray`.

        """
        mean_sky_level = get_mean_sky_level(survey, filt)
        blend_catalog.add_column(
            Column(np.zeros(len(blend_catalog)), name="not_drawn_" + filt.name)
        )
        pix_stamp_size = int(self.stamp_size / survey.pixel_scale)
        iso_image = np.zeros((self.max_number, pix_stamp_size, pix_stamp_size))
        _blend_image = galsim.Image(np.zeros((pix_stamp_size, pix_stamp_size)))

        for k, entry in enumerate(blend_catalog):
            single_image = self.render_single(entry, psf, filt, survey)
            iso_image[k] = single_image.array
            _blend_image += single_image

        # add background
        _blend_image += mean_sky_level

        # add noise.
        if self.add_noise:
            if self.verbose:
                print("Noise added to blend image")
            generator = galsim.random.BaseDeviate(seed=np.random.randint(99999999))
            noise = galsim.PoissonNoise(rng=generator)
            _blend_image.addNoise(noise)

        blend_image = _blend_image.array
        return blend_image, iso_image

    @abstractmethod
    def render_single(self, entry, psf, filt, survey):
        """Renders single galaxy in single band in the location given by its entry
        using the cutout information.

        The image created must be in a stamp of size stamp_size / cutout.pixel_scale.

        Return:
            galsim.Image
        """


class CatsimGenerator(DrawBlendsGenerator):
    compatible_catalogs = ("CatsimCatalog",)

    def render_single(self, entry, psf, filt, survey):
        """Returns the Galsim Image of an isolated galaxy.

        Args:
            entry: `descwl.model.Galaxy` class that models galaxies.
            cutout: `descwl.survey.Survey` class describing observing conditions.
                The input galaxy is rendered and stored in here.
            band:

        Return:
            galsim.Image object
        """
        if self.verbose:
            print("Draw isolated object")

        pix_stamp_size = int(self.stamp_size / survey.pixel_scale)
        try:
            gal = get_catsim_galaxy(entry, filt, survey)
            gal_conv = galsim.Convolve(gal, psf)
            gal_conv = gal_conv.shift(entry["ra"], entry["dec"])
            return gal_conv.drawImage(
                nx=pix_stamp_size, ny=pix_stamp_size, scale=survey.pixel_scale
            )

        except SourceNotVisible:
            if self.verbose:
                print("Source not visible")
            entry["not_drawn_" + filt.name] = 1


class CosmosGenerator(DrawBlendsGenerator):
    compatible_catalogs = ("CosmosCatalog",)

    def render_single(self, entry, psf, filt, survey):
        """Draws a single random galaxy profile in a random location of the image
        Args:
            shift (np.array): pixel center of the single galaxy in the postage stamp
                              with size self.stamp_size

        Returns:
            gal (galsim.InterpolatedImage): The galsim profile of a single galaxy
        """
        k = int(np.random.rand(1) * len(self.catalog))  # catalog_line["btk_index"][0]
        cat = self.catalog.get
        gal = self.catalog.makeGalaxy(k, gal_type="real", noise_pad_size=0).withFlux(1)
        pix_stamp_size = int(self.stamp_size / survey.pixel_scale)

        # Convolution by a smal gaussian: The galsim models actally have noise in a little patch around them,
        # so gaussian kernel convolution smoothes it out.
        # It has the slight disadvantage of adding some band-limitedeness to the image,
        # but with a small kernel, it's better than doing nothing.
        gal = galsim.Convolve(gal, galsim.Gaussian(sigma=2 * survey.pixel_scale))

        # Randomly shifts the galaxy in the patch
        galaxy = (
            galsim.Convolve(gal, psf)
            .drawImage(
                nx=pix_stamp_size,
                ny=pix_stamp_size,
                use_true_center=True,
                method="real_space",
                scale=survey.pixel_scale,
                dtype=np.float64,
            )
            .array
        )

        return galsim.Image(galaxy)
