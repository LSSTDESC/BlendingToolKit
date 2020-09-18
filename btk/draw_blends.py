import copy
from itertools import chain
from abc import ABC, abstractmethod

import descwl
import galsim
import numpy as np
from astropy.table import Column

from btk.multiprocess import multiprocess
from btk.obs_conditions import all_surveys


def get_center_in_pixels(blend_catalog, stamp_size, pixel_scale):
    """Returns center of objects in blend_catalog in pixel coordinates of
    postage stamp.

    blend_catalog contains ra dec of object center with the postage stamp
    center being 0,0. The size of the postage stamp and pixel scale is used to
    compute the object centers in pixel coordinates. Coordinates are in pixels
    where bottom left corner of postage stamp is (0, 0).

    Args:
        blend_catalog: Catalog with entries corresponding to one blend.
        stamp_size: In arcseconds.
        pixel_scale: Number of pixels per arcsecond.

    Returns:
        `astropy.table.Column`: x and y coordinates of object centroid
    """
    center = (stamp_size / pixel_scale - 1) / 2
    dx = blend_catalog["ra"] / pixel_scale + center
    dy = blend_catalog["dec"] / pixel_scale + center
    dx_col = Column(dx, name="dx")
    dy_col = Column(dy, name="dy")
    return dx_col, dy_col


def get_size(pixel_scale, catalog, meas_obs_cond):
    """Returns a astropy.table.column with the size of the galaxy.

    Galaxy size is estimated as second moments size (r_sec) computed as
    described in A1 of Chang et.al 2012. The PSF second moment size, psf_r_sec,
    is computed by galsim from the PSF model in obs_conds in the i band.
    The object size is the defined as sqrt(r_sec**2 + 2*psf_r_sec**2).

    Args:
        pixel_scale: arcseconds per pixel
        catalog: Catalog with entries corresponding to one blend.
        meas_obs_cond: `descwl.survey.Survey` class describing
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
    psf = meas_obs_cond.psf_model
    psf_r_sec = psf.calculateMomentRadius()
    size = np.sqrt(r_sec ** 2 + psf_r_sec ** 2) / pixel_scale
    return Column(size, name="size")


class DrawBlendsGenerator(ABC):
    def __init__(
        self,
        blend_generator,
        observing_generator,
        meas_band="i",
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
            observing_generator: Observing generator for each entry in batch.
            multiprocessing: Divides batch of blends to draw into mini-batches and
                runs each on different core
            cpus: If multiprocessing, then number of parallel processes to run.
            meas_band:
        """

        self.blend_generator = blend_generator
        self.observing_generator = observing_generator
        self.multiprocessing = multiprocessing
        self.cpus = cpus

        self.batch_size = self.blend_generator.batch_size
        self.max_number = self.blend_generator.max_number

        self.surveys = self.observing_generator.surveys
        self.stamp_size = self.observing_generator.obs_conds.stamp_size

        self.bands = {}
        for s in self.surveys:
            self.bands[s] = all_surveys[s]["bands"]
        self.meas_band = meas_band

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
            pix_stamp_size = int(self.stamp_size / all_surveys[s]["pixel_scale"])
            blend_images[s] = np.zeros(
                (self.batch_size, pix_stamp_size, pix_stamp_size, len(self.bands[s]))
            )
            isolated_images[s] = np.zeros(
                (
                    self.batch_size,
                    self.max_number,
                    pix_stamp_size,
                    pix_stamp_size,
                    len(self.bands[s]),
                )
            )
            batch_blend_cat[s], batch_obs_cond[s] = [], []

        in_batch_blend_cat = next(self.blend_generator)
        obs_conds = next(self.observing_generator)
        mini_batch_size = np.max([self.batch_size // self.cpus, 1])
        for s in self.surveys:
            input_args = [
                (
                    copy.deepcopy(in_batch_blend_cat[i : i + mini_batch_size]),
                    copy.deepcopy(obs_conds[s]),
                    s,
                )
                for i in range(0, self.batch_size, mini_batch_size)
            ]

            # multiprocess and join results
            mini_batch_results = multiprocess(
                self.run_mini_batch,
                input_args,
                self.cpus,
                self.multiprocessing,
                self.verbose,
            )
            batch_results = list(chain(*mini_batch_results))

            # organize results.
            for i in range(self.batch_size):
                blend_images[s][i] = batch_results[i][0]
                isolated_images[s][i] = batch_results[i][1]
                batch_blend_cat[s].append(batch_results[i][2])
        if len(self.surveys) > 1:
            output = {
                "blend_images": blend_images,
                "isolated_images": isolated_images,
                "blend_list": batch_blend_cat,
                "obs_condition": obs_conds,
            }
        else:
            survey_name = self.surveys[0]
            output = {
                "blend_images": blend_images[survey_name],
                "isolated_images": isolated_images[survey_name],
                "blend_list": batch_blend_cat[survey_name],
                "obs_condition": obs_conds,
            }
        return output

    @abstractmethod
    def run_mini_batch(self, blend_catalog, obs_conds):
        pass


class WLDGenerator(DrawBlendsGenerator):
    def draw_isolated(self, galaxy, iso_obs):
        """Returns `descwl.survey.Survey` class object that includes the rendered
        object for an isolated galaxy in its '.image' attribute.

        Args:
            galaxy: `descwl.model.Galaxy` class that models galaxies.
            iso_obs: `descwl.survey.Survey` class describing observing conditions.
                The input galaxy is rendered and stored in here.
        """
        if self.verbose:
            print("Draw isolated object")
        iso_render_engine = descwl.render.Engine(
            survey=iso_obs,
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
        return iso_obs

    def run_single_band(self, blend_catalog, obs_conds, band):
        """Draws image of isolated galaxies along with the blend image in the
        single input band.

        The WLDeblending package (descwl) renders galaxies corresponding to the
        blend_catalog entries and with observing conditions determined by
        obs_conds. The rendered objects are stored in the observing conditions
        class. So as to not overwrite images across different blends, we make a
        copy of the obs_conds while drawing each galaxy. Images of isolated
        galaxies are drawn with the WLDeblending and them summed to produce the
        blend image.

        A column 'not_drawn_{band}' is added to blend_catalog initialized as zero.
        If a galaxy was not drawn by descwl, then this flag is set to 1.

        Args:
            blend_catalog: Catalog with entries corresponding to one blend.
            obs_conds: `descwl.survey.Survey` class describing observing conditions.
            band(string): Name of band to draw images in.

        Returns:
            Images of blend and isolated galaxies as `numpy.ndarray`.

        """
        blend_catalog.add_column(
            Column(np.zeros(len(blend_catalog)), name="not_drawn_" + band)
        )
        galaxy_builder = descwl.model.GalaxyBuilder(
            obs_conds, no_disk=False, no_bulge=False, no_agn=False, verbose_model=False
        )
        pix_stamp_size = np.int(self.stamp_size / obs_conds.pixel_scale)
        iso_image = np.zeros((self.max_number, pix_stamp_size, pix_stamp_size))
        # define temporary galsim image
        # this will hold isolated galaxy images that will be summed
        blend_image_temp = galsim.Image(np.zeros((pix_stamp_size, pix_stamp_size)))
        mean_sky_level = obs_conds.mean_sky_level
        for k, entry in enumerate(blend_catalog):
            iso_obs = copy.deepcopy(obs_conds)
            try:
                galaxy = galaxy_builder.from_catalog(
                    entry, entry["ra"], entry["dec"], band
                )
                iso_render = self.draw_isolated(galaxy, iso_obs)
                iso_image[k] = iso_render.image.array
                blend_image_temp += iso_render.image
            except descwl.render.SourceNotVisible:
                if self.verbose:
                    print("Source not visible")
                blend_catalog["not_drawn_" + band][k] = 1
                continue
        if self.add_noise:
            if self.verbose:
                print("Noise added to blend image")
            generator = galsim.random.BaseDeviate(seed=np.random.randint(99999999))
            noise = galsim.PoissonNoise(rng=generator, sky_level=mean_sky_level)
            blend_image_temp.addNoise(noise)
        blend_image = blend_image_temp.array
        return blend_image, iso_image

    def run_mini_batch(self, blend_list, obs_conds, survey_name):
        """Returns isolated and blended images for bend catalogs in blend_list

        Function loops over blend_list and draws blend and isolated images in each
        band. Even though blend_list was input to the function, we return it since,
        the blend catalogs now include additional columns that flag if an object
        was not drawn and object centers in pixel coordinates.

        Args:
            blend_list: List of catalogs with entries corresponding to one blend.
            obs_conds (list): List of `descwl.survey.Survey` class describing
                observing conditions in different bands.

        Returns:
            `numpy.ndarray` of blend images and isolated galaxy images, along with
            list of blend catalogs.
        """
        mini_batch_outputs = []
        for i in range(len(blend_list)):
            pixel_scale = obs_conds[0].pixel_scale
            dx, dy = get_center_in_pixels(blend_list[i], self.stamp_size, pixel_scale)
            blend_list[i].add_column(dx)
            blend_list[i].add_column(dy)
            size = get_size(
                pixel_scale,
                blend_list[i],
                obs_conds[self.bands[survey_name] == self.meas_band],
            )
            blend_list[i].add_column(size)
            pix_stamp_size = int(self.stamp_size / pixel_scale)
            iso_image_multi = np.zeros(
                (
                    self.max_number,
                    pix_stamp_size,
                    pix_stamp_size,
                    len(self.bands[survey_name]),
                )
            )
            blend_image_multi = np.zeros(
                (pix_stamp_size, pix_stamp_size, len(self.bands[survey_name]))
            )
            for j in range(len(self.bands[survey_name])):
                single_band_output = self.run_single_band(
                    blend_list[i], obs_conds[j], self.bands[survey_name][j]
                )
                blend_image_multi[:, :, j] = single_band_output[0]
                iso_image_multi[:, :, :, j] = single_band_output[1]

            mini_batch_outputs.append(
                [blend_image_multi, iso_image_multi, blend_list[i]]
            )
        return mini_batch_outputs
