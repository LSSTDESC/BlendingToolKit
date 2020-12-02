import copy
from itertools import chain
from abc import ABC, abstractmethod

import galsim
import numpy as np
from astropy.table import Column

import descwl

from btk.multiprocess import multiprocess


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


def get_size(pixel_scale, catalog, cutout):
    """Returns a astropy.table.column with the size of the galaxy.

    Galaxy size is estimated as second moments size (r_sec) computed as
    described in A1 of Chang et.al 2012. The PSF second moment size, psf_r_sec,
    is computed by galsim from the PSF model in obs_conds in the i band.
    The object size is the defined as sqrt(r_sec**2 + 2*psf_r_sec**2).

    Args:
        pixel_scale: arcseconds per pixel
        catalog: Catalog with entries corresponding to one blend.
        cutout: `btk.Cutout.cutout` class describing
                observing conditions in bands to take measurement in.

    Returns:
        `astropy.table.Column`: size of the galaxy.
    """
    f = catalog["fluxnorm_bulge"] / (
        catalog["fluxnorm_disk"] + catalog["fluxnorm_bulge"]
    )
    hlr_d = np.sqrt(catalog["a_d"] * catalog["b_d"])
    hlr_b = np.sqrt(catalog["a_b"] * catalog["b_b"])
    r_sec = np.hypot(hlr_d * (1 - f) ** 0.5 * 4.66, hlr_b * f ** 0.5 * 1.46)
    psf = cutout.psf_model
    psf_r_sec = psf.calculateMomentRadius()
    size = np.sqrt(r_sec ** 2 + psf_r_sec ** 2) / pixel_scale
    return Column(size, name="size")


class DrawBlendsGenerator(ABC):

    compatible_catalogs = ("Catalog",)

    def __init__(
        self,
        blend_generator,
        observing_generator,
        meas_bands=("i",),
        multiprocessing=False,
        cpus=1,
        verbose=False,
        add_noise=True,
        min_snr=0.05,
    ):
        """Class that generates images of blended objects, individual isolated
        objects, for each blend in the batch.

        Batch is divided into mini batches of size blend_generator.batch_size//cpus and
        each mini-batch analyzed separately. The results are then combined to output a
        dict with results of entire batch. If multiprocessing is true, then each of
        the mini-batches are run in parallel.

        Args:
            blend_generator: Object generator to create blended object
            observing_generator: Observing generator to get observing conditions.
                                    The observing conditions are the same for the
                                    whole batch.
            multiprocessing: Divides batch of blends to draw into mini-batches and
                runs each on different core
            cpus: If multiprocessing, then number of parallel processes to run.
            meas_bands (tuple): For each survey in `self.observing_generator.surveys`,
                               the band in that survey for which measurements of e.g.
                               size will be made. Tuple order should be same as
                               `surveys`.
        """

        self.blend_generator = blend_generator
        self.observing_generator = observing_generator
        self.catalog = self.blend_generator.catalog
        self.multiprocessing = multiprocessing
        self.cpus = cpus

        self.batch_size = self.blend_generator.batch_size
        self.max_number = self.blend_generator.max_number

        self.surveys = self.observing_generator.surveys
        self.stamp_size = self.observing_generator.obs_conds.stamp_size

        self.meas_bands = {}
        for i, s in enumerate(self.surveys):
            self.meas_bands[s["name"]] = meas_bands[i]

        self.add_noise = add_noise
        self.min_snr = min_snr
        self.verbose = verbose

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
        for s in self.surveys:
            pix_stamp_size = int(self.stamp_size / s["pixel_scale"])
            batch_blend_cat[s["name"]], batch_obs_cond[s["name"]] = [], []
            blend_images[s["name"]] = np.zeros(
                (self.batch_size, pix_stamp_size, pix_stamp_size, len(s["bands"]))
            )
            isolated_images[s["name"]] = np.zeros(
                (
                    self.batch_size,
                    self.max_number,
                    pix_stamp_size,
                    pix_stamp_size,
                    len(s["bands"]),
                )
            )

        in_batch_blend_cat = next(self.blend_generator)
        obs_conds = next(self.observing_generator)  # same for every blend in batch.
        mini_batch_size = np.max([self.batch_size // self.cpus, 1])
        for s in self.surveys:
            input_args = [
                (
                    copy.deepcopy(in_batch_blend_cat[i : i + mini_batch_size]),
                    copy.deepcopy(obs_conds[s["name"]]),
                    s,
                )
                for i in range(0, self.batch_size, mini_batch_size)
            ]

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
                blend_images[s["name"]][i] = batch_results[i][0]
                isolated_images[s["name"]][i] = batch_results[i][1]
                batch_blend_cat[s["name"]].append(batch_results[i][2])
        if len(self.surveys) > 1:
            output = {
                "blend_images": blend_images,
                "isolated_images": isolated_images,
                "blend_list": batch_blend_cat,
                "obs_condition": obs_conds,
            }
        else:
            survey_name = self.surveys[0]["name"]
            output = {
                "blend_images": blend_images[survey_name],
                "isolated_images": isolated_images[survey_name],
                "blend_list": batch_blend_cat[survey_name],
                "obs_condition": obs_conds[survey_name],
            }
        return output

    def render_mini_batch(self, blend_list, cutouts, survey):
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
        for i in range(len(blend_list)):

            # All bands in same survey have same pixel scale, WCS
            pixel_scale = survey["pixel_scale"]
            wcs = cutouts[0].wcs

            # Band to do measurements of size for given survey.
            meas_band = self.meas_bands[survey["name"]]

            dx, dy = get_center_in_pixels(blend_list[i], wcs)
            blend_list[i].add_column(dx)
            blend_list[i].add_column(dy)
            # TODO: How to get size for COSMOS?
            size = get_size(
                pixel_scale,
                blend_list[i],
                cutouts[survey["bands"] == meas_band],
            )
            blend_list[i].add_column(size)
            pix_stamp_size = int(self.stamp_size / pixel_scale)
            iso_image_multi = np.zeros(
                (
                    self.max_number,
                    pix_stamp_size,
                    pix_stamp_size,
                    len(survey["bands"]),
                )
            )
            blend_image_multi = np.zeros(
                (pix_stamp_size, pix_stamp_size, len(survey["bands"]))
            )
            for j in range(len(survey["bands"])):
                single_band_output = self.render_blend(
                    blend_list[i], cutouts[j], survey["bands"][j]
                )
                blend_image_multi[:, :, j] = single_band_output[0]
                iso_image_multi[:, :, :, j] = single_band_output[1]

            outputs.append([blend_image_multi, iso_image_multi, blend_list[i]])
        return outputs

    def render_blend(self, blend_catalog, cutout, band):
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
            cutout: `btk.obs_conditions.Cutout` class describing observing conditions.
            band(string): Name of band to draw images in.

        Returns:
            Images of blend and isolated galaxies as `numpy.ndarray`.

        """
        if not hasattr(cutout, "mean_sky_level"):
            raise AttributeError("cutout needs a mean_sky_level as an attribute.")

        blend_catalog.add_column(
            Column(np.zeros(len(blend_catalog)), name="not_drawn_" + band)
        )

        pix_stamp_size = np.int(self.stamp_size / cutout.pixel_scale)
        iso_image = np.zeros((self.max_number, pix_stamp_size, pix_stamp_size))

        # define galsim image
        _blend_image = galsim.Image(np.zeros((pix_stamp_size, pix_stamp_size)))
        mean_sky_level = cutout.mean_sky_level

        for k, entry in enumerate(blend_catalog):
            try:
                _cutout = copy.deepcopy(cutout)
                single_image = self.render_single(entry, _cutout, band)
                if single_image.array.shape[-1] != pix_stamp_size:
                    raise ValueError(
                        "render_single returned image of incorrect dimensions."
                    )

                iso_image[k] = single_image.array
                _blend_image += single_image

            except descwl.render.SourceNotVisible:
                continue

        if self.add_noise:
            if self.verbose:
                print("Noise added to blend image")
            generator = galsim.random.BaseDeviate(seed=np.random.randint(99999999))
            noise = galsim.PoissonNoise(rng=generator, sky_level=mean_sky_level)
            _blend_image.addNoise(noise)

        blend_image = _blend_image.array
        return blend_image, iso_image

    @abstractmethod
    def render_single(self, entry, cutout, band):
        """Renders single galaxy in single band in the location given by its entry
        using the cutout information.

        The image created must be in a stamp of size stamp_size / cutout.pixel_scale.

        Return:
            galsim.Image
        """
        pass


class WLDGenerator(DrawBlendsGenerator):
    compatible_catalogs = ("WLDCatalog",)

    def render_single(self, entry, cutout, band):
        """Returns the Galsim Image of an isolated galaxy.

        Args:
            entry: `descwl.model.Galaxy` class that models galaxies.
            cutout: `descwl.survey.Survey` class describing observing conditions.
                The input galaxy is rendered and stored in here.
            band:

        Return:
            galsim.Image
        """
        if self.verbose:
            print("Draw isolated object")

        galaxy_builder = descwl.model.GalaxyBuilder(
            cutout, no_disk=False, no_bulge=False, no_agn=False, verbose_model=False
        )

        try:
            galaxy = galaxy_builder.from_catalog(entry, entry["ra"], entry["dec"], band)
            iso_render_engine = descwl.render.Engine(
                survey=cutout,
                min_snr=self.min_snr,
                truncate_radius=30,
                no_margin=False,
                verbose_render=False,
            )
            iso_render_engine.render_galaxy(
                galaxy,
                variations_x=None,
                variations_s=None,
                variations_g=None,
                no_fisher=True,
                no_analysis=True,
            )

            return cutout.image

        except descwl.render.SourceNotVisible:
            if self.verbose:
                print("Source not visible")
            entry["not_drawn_" + band] = 1
            raise descwl.render.SourceNotVisible


class GalsimRealDraw(DrawBlendsGenerator):
    """Class that instantiates a blend from real galsim images
    Parameters
    ----------
    cat:
        galsim catalog of galaxies
    pix: `float`
        pixel scale in arcseconds
    stamp_size: `int`
        number of pixels on a side for the stamp
    channels: `array`
        array of names for each band in the blend
    sky_center: `array`
        coordinate of a reference pixel in ra-dec
    pixe_center: `array`
        pixel coordinates of a reference pixel
    psf_function: function
        function to generate a 2-dimensional psf profile
    psf_args: `list`
        list of arguments for the psf function
    """

    compatible_catalogs = ("CosmosCatalog",)

    def __init__(
        self,
        cat,
        pix,
        stamp_size,
    ):
        self.cat = cat
        self.pix = pix
        self.stamp_size = stamp_size

        self.seds = None
        self.singles = None
        self.locs = None
        self.blend = None

    def draw_single(self, cat, shift):
        """Draws a single random galaxy profile in a random location of the image
        Args:
            shift (np.array): pixel center of the single galaxy in the postage stamp
                              with size self.stamp_size

        Returns:
            gal (galsim.InterpolatedImage): The galsim profile of a single galaxy
        """

        k = np.int(np.random.randn(1) * cat.size)
        gal = cat.makeGalaxy(
            k, gal_type="real", noise_pad_size=self.stamp_size * self.pixel_scale
        )
        gal = gal.shift(dx=shift[0], dy=shift[1])
        return gal

    def draw_blend(self, ngal):
        """Creates multi-band scenes
        Args:
            ngal (int): Number of galaxies in the stamp
        """
        singles = []
        seds = []
        locs = []
        cube = np.zeros((self.channels.size, self.stamp_size, self.stamp_size))
        for i in range(ngal):
            shift = (np.random.rand(2) - 0.5) * self.stamp_size * self.pix / 2
            gal = self.draw_single(shift)
            singles.append(gal)
            sed = np.random.rand(self.channels.size) * 0.8 + 0.2
            seds.appends(sed)

            im = galsim.Convolve(gal, self.psf).drawImage(
                nx=self.stamp_size,
                ny=self.stamp_size,
                use_true_center=True,
                method="no_pixel",
                scale=self.pix,
                dtype=np.float64,
            )
            locs.append(
                [
                    shift[0] / self.pix + self.stamp_size[0] / 2,
                    shift[1] / self.pix + self.stamp_size[1] / 2,
                ]
            )
            cube += im[None, :, :] * sed[:, None, None]

        self.singles = singles
        self.locs = locs
        self.seds = seds

        return cube

    def render_single(self, entry, cutout, band):
        pass


class GalsimHubDraw(DrawBlendsGenerator):
    def render_single(self, entry, cutout, band):
        pass
