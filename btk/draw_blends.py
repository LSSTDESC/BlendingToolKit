"""Module for generating batches of drawn blended images."""
import copy
import os
from abc import ABC
from abc import abstractmethod
from itertools import chain

import galsim
import numpy as np
from astropy.table import Column
from astropy.table import Table

from btk.create_blend_generator import BlendGenerator
from btk.multiprocess import multiprocess
from btk.survey import get_flux
from btk.survey import get_mean_sky_level
from btk.survey import make_wcs
from btk.survey import Survey


class SourceNotVisible(Exception):
    """Custom exception to indicate that a source has no visible model components."""


def get_center_in_pixels(blend_catalog, wcs):
    """Returns center of objects in blend_catalog in pixel coordinates of postage stamp.

    `blend_catalog` contains `ra, dec` of object center with the postage stamp
    center being 0,0. The size of the postage stamp and pixel scale is used to
    compute the object centers in pixel coordinates. Coordinates are in pixels
    where bottom left corner of postage stamp is (0, 0).

    Args:
        blend_catalog: Catalog with entries corresponding to one blend.
        wcs (astropy.wcs.WCS): astropy WCS object corresponding to the image
    Returns:
        `astropy.table.Column`: x and y coordinates of object centroid
    """
    x_peak, y_peak = wcs.world_to_pixel_values(
        blend_catalog["ra"] / 3600, blend_catalog["dec"] / 3600
    )
    dx_col = Column(x_peak, name="x_peak")
    dy_col = Column(y_peak, name="y_peak")
    return dx_col, dy_col


def get_catsim_galaxy(entry, filt, survey, no_disk=False, no_bulge=False, no_agn=False):
    """Returns a bulge/disk/agn Galsim galaxy profile based on entry.

    This function returns a composite galsim galaxy profile with bulge, disk and AGN based on the
    parameters in entry, and using observing conditions defined by the survey
    and the filter.

    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        entry (astropy.table.Table): single astropy line containing information on the galaxy
        survey (btk.survey.Survey): BTK Survey object
        filt (btk.survey.Filter): BTK Filter object
        no_disk (bool): Sets the flux for the disk to zero
        no_bulge (bool): Sets the flux for the bulge to zero
        no_agn (bool): Sets the flux for the AGN to zero
    Returns:
        Galsim galaxy profile
    """
    components = []
    total_flux = get_flux(entry[filt.name + "_ab"], filt, survey)
    # Calculate the flux of each component in detected electrons.
    total_fluxnorm = entry["fluxnorm_disk"] + entry["fluxnorm_bulge"] + entry["fluxnorm_agn"]
    disk_flux = 0.0 if no_disk else entry["fluxnorm_disk"] / total_fluxnorm * total_flux
    bulge_flux = 0.0 if no_bulge else entry["fluxnorm_bulge"] / total_fluxnorm * total_flux
    agn_flux = 0.0 if no_agn else entry["fluxnorm_agn"] / total_fluxnorm * total_flux

    if disk_flux + bulge_flux + agn_flux == 0:
        raise SourceNotVisible

    if disk_flux > 0:
        beta_radians = np.radians(entry["pa_disk"])
        if bulge_flux > 0:
            assert entry["pa_disk"] == entry["pa_bulge"], "Sersic components have different beta."
        a_d, b_d = entry["a_d"], entry["b_d"]
        disk_hlr_arcsecs = np.sqrt(a_d * b_d)
        disk_q = b_d / a_d
        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            q=disk_q, beta=beta_radians * galsim.radians
        )
        components.append(disk)

    if bulge_flux > 0:
        beta_radians = np.radians(entry["pa_bulge"])
        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
        bulge_q = b_b / a_b
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
            q=bulge_q, beta=beta_radians * galsim.radians
        )
        components.append(bulge)

    if agn_flux > 0:
        agn = galsim.Gaussian(flux=agn_flux, sigma=1e-8)
        components.append(agn)

    profile = galsim.Add(components)
    return profile


class DrawBlendsGenerator(ABC):
    """Class that generates images of blends and individual isolated objects in batches.

    Batch is divided into mini batches of size blend_generator.batch_size//cpus and
    each mini-batch analyzed separately. The results are then combined to output a
    dict with results of entire batch. If the number of cpus is greater than one, then each of
    the mini-batches are run in parallel.

    """

    compatible_catalogs = ("Catalog",)

    def __init__(
        self,
        catalog,
        sampling_function,
        surveys: list,
        batch_size=8,
        stamp_size=24,
        cpus=1,
        verbose=False,
        add_noise=True,
        shifts=None,
        indexes=None,
        channels_last=False,
        save_path=None,
        rng=None,
    ):
        """Initializes the DrawBlendsGenerator class.

        Args:
            catalog (btk.catalog.Catalog): BTK catalog object from which galaxies are taken.
            sampling_function (btk.sampling_function.SamplingFunction): BTK sampling
                function to use.
            surveys (list): List of btk Survey objects defining the observing conditions
            batch_size (int): Number of blends generated per batch
            stamp_size (float): Size of the stamps, in arcseconds
            cpus (int): Number of cpus to use; defines the number of minibatches
            verbose (bool): Indicates whether additionnal information should be printed
            add_noise (bool): Indicates if the blends should be generated with noise
            shifts (list): Contains arbitrary shifts to be applied instead of
                           random shifts. Must be of length batch_size. Must be used
                           with indexes.
            indexes (list): Contains the ids of the galaxies to use in the stamp.
                        Must be of length batch_size. Must be used with shifts.
            channels_last (bool): Whether to return images as numpy arrays with the channel
                                (band) dimension as the last dimension or before the pixels
                                dimensions (default).
            save_path (str): Path to a directory where results will be saved. If left
                            as None, results will not be saved.
            rng : Controls the random number generation. Can be an integer seed,
                  or a numpy.random.Generator. If None, a random seed will be used.
        """
        self.blend_generator = BlendGenerator(
            catalog, sampling_function, batch_size, shifts, indexes, verbose
        )
        self.catalog = self.blend_generator.catalog
        self.cpus = cpus

        self.batch_size = self.blend_generator.batch_size
        self.max_number = self.blend_generator.max_number

        if isinstance(surveys, Survey):
            self.surveys = [surveys]
            self.check_compatibility(surveys)
        elif isinstance(surveys, list):
            for s in surveys:
                if not isinstance(s, Survey):
                    raise TypeError("surveys must be a Survey object or a list of Survey objects.")
                self.check_compatibility(s)
            self.surveys = surveys
        else:
            raise TypeError("surveys must be a Survey object or a list of Survey objects.")
        self.is_multiresolution = len(self.surveys) > 1

        self.stamp_size = stamp_size
        self.add_noise = add_noise
        self.verbose = verbose
        self.channels_last = channels_last
        self.save_path = save_path

        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        else:
            try:
                rng.random()
            except AttributeError:
                raise AttributeError("The random generator you provided is invalid.")
            self.rng = rng

    def check_compatibility(self, survey):
        """Checks that the compatibility between the survey, the catalog and the generator.

        This should be implemented in subclasses.
        """
        pass

    def __iter__(self):
        """Returns iterable which is the object itself."""
        return self

    def __next__(self):
        """Outputs dictionary containing blend output (images and catalogs) in batches.

        Returns:
            output: Dictionary with blend images, isolated object images, blend catalog,
            PSF images and WCS.
        """
        blend_list = {}
        blend_images = {}
        isolated_images = {}
        blend_cat = next(self.blend_generator)
        mini_batch_size = np.max([self.batch_size // self.cpus, 1])
        psfs = {}
        wcss = {}

        for s in self.surveys:
            pix_stamp_size = int(self.stamp_size / s.pixel_scale)

            # make PSF and WCS
            psf = []
            for filt in s.filters:
                if callable(filt.psf):
                    generated_psf = filt.psf()  # generate the PSF with the provided function
                    if isinstance(generated_psf, galsim.GSObject):
                        psf.append(generated_psf)
                    else:
                        raise TypeError(
                            f"The generated PSF with the provided function"
                            f"for filter '{filt.name}' is not a galsim object"
                        )
                elif isinstance(filt.psf, galsim.GSObject):
                    psf.append(filt.psf)  # or directly retrieve the PSF
                else:
                    raise TypeError(
                        f"The PSF within filter '{filt.name}' is neither a "
                        f"function nor a galsim object"
                    )
            wcs = make_wcs(s.pixel_scale, (pix_stamp_size, pix_stamp_size))
            psfs[s.name] = psf
            wcss[s.name] = wcs

            input_args = []
            for i in range(0, self.batch_size, mini_batch_size):
                cat = copy.deepcopy(blend_cat[i : i + mini_batch_size])
                input_args.append((cat, psf, wcs, s))

            # multiprocess and join results
            # ideally, each cpu processes a single mini_batch
            mini_batch_results = multiprocess(
                self.render_mini_batch,
                input_args,
                cpus=self.cpus,
                verbose=self.verbose,
            )

            # join results across mini-batches.
            batch_results = list(chain(*mini_batch_results))

            # decide image_shape based on channels_last bool.
            option1 = (len(s.filters), pix_stamp_size, pix_stamp_size)
            option2 = (pix_stamp_size, pix_stamp_size, len(s.filters))
            image_shape = option1 if not self.channels_last else option2

            # organize results.
            blend_images[s.name] = np.zeros((self.batch_size, *image_shape))
            isolated_images[s.name] = np.zeros((self.batch_size, self.max_number, *image_shape))
            blend_list[s.name] = []
            for i in range(self.batch_size):
                blend_images[s.name][i] = batch_results[i][0]
                isolated_images[s.name][i] = batch_results[i][1]
                blend_list[s.name].append(batch_results[i][2])

            # save results if requested.
            if self.save_path is not None:
                if not os.path.exists(os.path.join(self.save_path, s.name)):
                    os.mkdir(os.path.join(self.save_path, s.name))

                np.save(os.path.join(self.save_path, s.name, "blended"), blend_images[s.name])
                np.save(os.path.join(self.save_path, s.name, "isolated"), isolated_images[s.name])
                for i in range(len(batch_results)):
                    blend_list[s.name][i].write(
                        os.path.join(self.save_path, s.name, f"blend_info_{i}"),
                        format="ascii",
                        overwrite=True,
                    )
        if self.is_multiresolution:
            output = {
                "blend_images": blend_images,
                "isolated_images": isolated_images,
                "blend_list": blend_list,
                "psf": psfs,
                "wcs": wcss,
            }
        else:
            survey_name = self.surveys[0].name
            output = {
                "blend_images": blend_images[survey_name],
                "isolated_images": isolated_images[survey_name],
                "blend_list": blend_list[survey_name],
                "psf": psfs[survey_name],
                "wcs": wcss[survey_name],
            }
        return output

    def render_mini_batch(self, blend_list, psf, wcs, survey, extra_data=None):
        """Returns isolated and blended images for blend catalogs in blend_list.

        Function loops over blend_list and draws blend and isolated images in each
        band. Even though blend_list was input to the function, we return it since,
        the blend catalogs now include additional columns that flag if an object
        was not drawn and object centers in pixel coordinates.

        Args:
            blend_list (list): List of catalogs with entries corresponding to one
                               blend. The size of this list is equal to the
                               mini_batch_size.
            psf (list): List of Galsim objects containing the PSF
            wcs (astropy.wcs.WCS): astropy WCS object
            survey (dict): Dictionary containing survey information.
            extra_data: This field can be used if some data need to be generated
                before getting to the step where single galaxies are drawn. It should
                have a "shape" of (batch_size,n_blend,...) where n_blend is the number
                of objects in a blend. See GalsimHubGenerator for an example of usage.

        Returns:
            `numpy.ndarray` of blend images and isolated galaxy images, along with
            list of blend catalogs.
        """
        outputs = []
        index = 0

        if extra_data is None:
            extra_data = np.zeros((len(blend_list), np.max([len(blend) for blend in blend_list])))
        for i, blend in enumerate(blend_list):

            # All bands in same survey have same pixel scale, WCS
            pixel_scale = survey.pixel_scale
            pix_stamp_size = int(self.stamp_size / pixel_scale)

            x_peak, y_peak = get_center_in_pixels(blend, wcs)
            blend.add_column(x_peak)
            blend.add_column(y_peak)

            iso_image_multi = np.zeros(
                (
                    self.max_number,
                    len(survey.filters),
                    pix_stamp_size,
                    pix_stamp_size,
                )
            )
            blend_image_multi = np.zeros((len(survey.filters), pix_stamp_size, pix_stamp_size))
            for b, filt in enumerate(survey.filters):
                single_band_output = self.render_blend(blend, psf[b], filt, survey, extra_data[i])
                blend_image_multi[b, :, :] = single_band_output[0]
                iso_image_multi[:, b, :, :] = single_band_output[1]

            # transpose if requested.
            dim_order = np.array((0, 1, 2) if not self.channels_last else (1, 2, 0))
            blend_image_multi = blend_image_multi.transpose(dim_order)
            iso_image_multi = iso_image_multi.transpose(0, *(dim_order + 1))

            outputs.append([blend_image_multi, iso_image_multi, blend])
            index += len(blend)
        return outputs

    def render_blend(self, blend_catalog, psf, filt, survey, extra_data):
        """Draws image of isolated galaxies along with the blend image in the single input band.

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
            blend_catalog (astropy.table.Table): Catalog with entries corresponding to one blend.
            psf: Galsim object containing the psf for the given filter
            filt (btk.survey.Filter): BTK Filter object
            survey (btk.survey.Survey): BTK Survey object
            extra_data: Special field of shape (n_blend,?), containing
                additional data for drawing the blend. See render_minibatch
                method for more details.

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
            single_image = self.render_single(entry, filt, psf, survey, extra_data[k])
            iso_image[k] = single_image.array
            _blend_image += single_image

        # add noise.
        if self.add_noise:
            if self.verbose:
                print("Noise added to blend image")
            generator = galsim.random.BaseDeviate(seed=self.rng.integers(100000))
            noise = galsim.PoissonNoise(rng=generator, sky_level=mean_sky_level)
            _blend_image.addNoise(noise)

        blend_image = _blend_image.array
        return blend_image, iso_image

    @abstractmethod
    def render_single(self, entry, filt, psf, survey, extra_data):
        """Renders single galaxy in single band in the location given by its entry.

        The image created must be in a stamp of size stamp_size / cutout.pixel_scale. The image
        must be drawn according to information provided by filter, psf, and survey.

        Args:
            entry (astropy.table.Table): Line from astropy describing the galaxy to draw
            filt (btk.survey.Filter): BTK Filter object corresponding to the band where
                                the image is drawn `
            psf: Galsim object containing the PSF relative to the chosen filter
            survey (btk.survey.Survey): BTK Survey object
            extra_data: Special field containing extra data for drawing a single galaxy.
                See render_minibatch method for more details.

        Returns:
            galsim.Image object
        """


class CatsimGenerator(DrawBlendsGenerator):
    """Implementation of DrawBlendsGenerator for drawing galaxies from a Catsim-like catalog.

    The code for drawing these galaxies and the default PSF is taken almost directly from
    WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending).
    """

    compatible_catalogs = ("CatsimCatalog",)

    def check_compatibility(self, survey):
        """Checks the compatibility between the catalog and a given survey.

        Args:
            survey (btk.survey.Survey): Survey to check
        """
        if type(self.catalog).__name__ not in self.compatible_catalogs:
            raise ValueError(
                f"The catalog provided is of the wrong type. The types of "
                f"catalogs available for the {type(self).__name__} are {self.compatible_catalogs}"
            )
        for f in survey.filters:
            if f.name + "_ab" not in self.catalog.table.keys():
                raise ValueError(
                    f"The {f.name} filter of the survey {survey.name} "
                    f"has no associated magnitude in the given catalog."
                )

    def render_single(self, entry, filt, psf, survey, extra_data):
        """Returns the Galsim Image of an isolated galaxy."""
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
    """Subclass of DrawBlendsGenerator for drawing real galaxies from the COSMOS catalog."""

    compatible_catalogs = ("CosmosCatalog",)

    def check_compatibility(self, survey):
        """Checks the compatibility between the catalog and a given survey.

        Args:
            survey (btk.survey.Survey): Survey to check
        """
        if type(self.catalog).__name__ not in self.compatible_catalogs:
            raise ValueError(
                f"The catalog provided is of the wrong type. The types of "
                f"catalogs available for the {type(self).__name__} are {self.compatible_catalogs}"
            )
        if "ref_mag" not in self.catalog.table.keys():
            for f in survey.filters:
                if f"{survey.name}_{f.name}" not in self.catalog.table.keys():
                    raise ValueError(
                        f"The {f.name} filter of the survey {survey.name} "
                        f"has no associated magnitude in the given catalog, "
                        f"and the catalog does not contain a 'ref_mag' column"
                    )

    def render_single(self, entry, filt, psf, survey, extra_data):
        """Returns the Galsim Image of an isolated galaxy."""
        galsim_catalog = self.catalog.get_galsim_catalog()

        # get galaxy flux
        try:
            gal_flux = get_flux(entry[f"{survey.name}_{filt.name}"], filt, survey)
        except KeyError:
            gal_flux = get_flux(entry["ref_mag"], filt, survey)

        gal = galsim_catalog.makeGalaxy(
            entry["btk_index"], gal_type="real", noise_pad_size=0
        ).withFlux(gal_flux)

        pix_stamp_size = int(self.stamp_size / survey.pixel_scale)

        # Convolve the galaxy with the PSF
        gal_conv = galsim.Convolve(gal, psf)
        # Apply the shift
        gal_conv = gal_conv.shift(entry["ra"], entry["dec"])

        return gal_conv.drawImage(nx=pix_stamp_size, ny=pix_stamp_size, scale=survey.pixel_scale)


class GalsimHubGenerator(DrawBlendsGenerator):
    """Implementation of DrawBlendsGenerator for drawing galaxies simulated with galsim_hub.

    Galsim Hub (https://github.com/McWilliamsCenter/galsim_hub) is a framework
    for generating real-looking galaxies using deep learning models.
    """

    compatible_catalogs = ("CosmosCatalog",)

    def check_compatibility(self, survey):
        """Checks the compatibility between the catalog and a given survey.

        Args:
            survey (btk.survey.Survey): Survey to check
        """
        if type(self.catalog).__name__ not in self.compatible_catalogs:
            raise ValueError(
                f"The catalog provided is of the wrong type. The types of "
                f"catalogs available for the {type(self).__name__} are {self.compatible_catalogs}"
            )

    def __init__(
        self,
        catalog,
        sampling_function,
        surveys: list,
        batch_size=8,
        stamp_size=24,
        cpus=1,
        verbose=False,
        add_noise=True,
        shifts=None,
        indexes=None,
        channels_last=False,
        galsim_hub_model="hub:Lanusse2020",
        param_names=["flux_radius", "mag_auto", "zphot"],
        save_path=None,
        rng=None,
    ):  # noqa: D417
        """Initializes the GalsimHubGenerator class.

        Args:
            galsim_hub_model (str): Source of the model to use. Can be
                    either a distant model or a local one, see the
                    galsim_hub repo for more information.
            param_names (list): list of the parameters with which
                    the generation is parametrized; this is unique to
                    each model.
        """
        super().__init__(
            catalog,
            sampling_function,
            surveys,
            batch_size=batch_size,
            stamp_size=stamp_size,
            cpus=cpus,
            verbose=verbose,
            add_noise=add_noise,
            shifts=shifts,
            indexes=indexes,
            channels_last=channels_last,
            save_path=save_path,
            rng=rng,
        )
        import galsim_hub

        self.galsim_hub_model = galsim_hub.GenerativeGalaxyModel(galsim_hub_model)
        self.param_names = param_names

    def render_mini_batch(self, blend_list, psf, wcs, survey):
        """Returns isolated and blended images for blend catalogs in blend_list.

        Here we generate the images for all galaxies in the batch at the same
        time, since galsim_hub is optimized for batch generation.
        """
        galsim_hub_params = Table()
        for p in self.param_names:
            column = Column(np.concatenate([blend[p] for blend in blend_list]), p)
            galsim_hub_params.add_column(column)

        base_images = self.galsim_hub_model.sample(galsim_hub_params)
        base_images_l = []
        index = 0
        for blend in blend_list:
            base_images_l.append(base_images[index : index + len(blend)])
            index += len(blend)

        return super().render_mini_batch(blend_list, psf, wcs, survey, base_images_l)

    def render_single(self, entry, filt, psf, survey, extra_data):
        """Returns the Galsim Image of an isolated galaxy."""
        base_image = extra_data
        base_image = galsim.Convolve(base_image, psf)
        gal_flux = get_flux(entry["ref_mag"], filt, survey)
        base_image = base_image.withFlux(gal_flux)
        base_image = base_image.shift(entry["ra"], entry["dec"])

        pix_stamp_size = int(self.stamp_size / survey.pixel_scale)
        galaxy_image = base_image.drawImage(
            nx=pix_stamp_size, ny=pix_stamp_size, scale=survey.pixel_scale, dtype=np.float64
        )
        return galaxy_image


available_draw_blends = {
    "CatsimGenerator": CatsimGenerator,
    "CosmosGenerator": CosmosGenerator,
    "GalsimHubGenerator": GalsimHubGenerator,
}
