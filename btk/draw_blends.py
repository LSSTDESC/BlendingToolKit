"""Module for generating batches of drawn blended images."""
import copy
import json
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import chain
from typing import Dict, List

import galsim
import numpy as np
from astropy.table import Column, Table
from galcheat.utilities import mag2counts, mean_sky_level
from tqdm.auto import tqdm

from btk import DEFAULT_SEED
from btk.create_blend_generator import BlendGenerator
from btk.multiprocess import get_current_process, multiprocess
from btk.survey import Survey, get_surveys, make_wcs

MAX_SEED_INT = 1_000_000_000


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
    total_flux = mag2counts(entry[filt.name + "_ab"], survey, filt).to_value("electron")
    # Calculate the flux of each component in detected electrons.
    total_fluxnorm = entry["fluxnorm_disk"] + entry["fluxnorm_bulge"] + entry["fluxnorm_agn"]
    disk_flux = 0.0 if no_disk else entry["fluxnorm_disk"] / total_fluxnorm * total_flux
    bulge_flux = 0.0 if no_bulge else entry["fluxnorm_bulge"] / total_fluxnorm * total_flux
    agn_flux = 0.0 if no_agn else entry["fluxnorm_agn"] / total_fluxnorm * total_flux

    if disk_flux + bulge_flux + agn_flux == 0:
        raise SourceNotVisible

    if disk_flux > 0:
        if bulge_flux > 0:
            assert entry["pa_disk"] == entry["pa_bulge"], "Sersic components have different beta."
        a_d, b_d = entry["a_d"], entry["b_d"]
        disk_hlr_arcsecs = np.sqrt(a_d * b_d)
        disk_q = b_d / a_d
        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            q=disk_q, beta=entry["pa_disk"] * galsim.degrees
        )
        components.append(disk)

    if bulge_flux > 0:
        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
        bulge_q = b_b / a_b
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
            q=bulge_q, beta=entry["pa_bulge"] * galsim.degrees
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
        add_noise="all",
        shifts=None,
        indexes=None,
        channels_last=False,
        save_path=None,
        seed=DEFAULT_SEED,
        apply_shear=False,
        augment_data=False,
    ):
        """Initializes the DrawBlendsGenerator class.

        Args:
            catalog (btk.catalog.Catalog): BTK catalog object from which galaxies are taken.
            sampling_function (btk.sampling_function.SamplingFunction): BTK sampling
                function to use.
            surveys (list or btk.survey.Survey): List of BTK Survey objects or
                single BTK Survey object.
            batch_size (int): Number of blends generated per batch
            stamp_size (float): Size of the stamps, in arcseconds
            cpus (int): Number of cpus to use; defines the number of minibatches
            verbose (bool): Indicates whether additionnal information should be printed
            add_noise (str): Indicates if the blends should be generated with noise.
                            "all" indicates that all the noise should be applied,
                            "background" adds only the background noise,
                            "galaxy" only the galaxy noise, and "none" gives noiseless
                            images.
            shifts (list): Contains arbitrary shifts to be applied instead of
                           random shifts. Must be of length batch_size. Must be used
                           with indexes. Used mostly for internal testing purposes.
            indexes (list): Contains the ids of the galaxies to use in the stamp.
                        Must be of length batch_size. Must be used with shifts.
                        Used mostly for internal testing purposes.
            channels_last (bool): Whether to return images as numpy arrays with the channel
                                (band) dimension as the last dimension or before the pixels
                                dimensions (default).
            save_path (str): Path to a directory where results will be saved. If left
                            as None, results will not be saved.
            seed (int): Integer seed for reproducible random noise realizations.
            apply_shear (float): Whether to apply the shear specified in catalogs to galaxies.
                                If set to True, sampling function must add 'g1', 'g2' columns.
            augment_data (float): If set to True, augment data by adding a random rotation to every
                                galaxy drawn. Rotation added is proapaged to the `pa_bulge`
                                and `pa_disk` columns if using the `CatsimGenerator`. It is also
                                stored in the `btk_rotation` column.
        """
        self.blend_generator = BlendGenerator(
            catalog, sampling_function, batch_size, shifts, indexes, verbose
        )
        self.catalog = self.blend_generator.catalog
        self.cpus = cpus
        self.batch_size = self.blend_generator.batch_size
        self.max_number = self.blend_generator.max_number
        self.apply_shear = apply_shear
        self.augment_data = augment_data
        self.is_multiresolution = len(self.surveys) > 1
        self.stamp_size = stamp_size
        self._set_surveys(surveys)
        self.is_multiresolution = len(self.surveys) > 1

        noise_options = {"none", "all", "background", "galaxy"}
        if add_noise not in noise_options:
            raise ValueError(
                f"The options for add_noise are {noise_options}, but you provided {add_noise}"
            )
        self.add_noise = add_noise
        self.verbose = verbose
        self.channels_last = channels_last
        self.save_path = save_path
        self.seedseq = np.random.SeedSequence(seed)

    def _set_surveys(self, surveys):
        """Check if passed in argument `surveys` has correct format."""
        if isinstance(surveys, Survey):
            self.surveys = [surveys]
            self.check_compatibility(surveys)
        elif isinstance(surveys, Iterable):
            for s in surveys:
                if not isinstance(s, Survey):
                    raise TypeError(
                        f"surveys must be a Survey object or an Iterable of Survey objects, but "
                        f"Iterable contained object of type {type(s)}"
                    )
                self.check_compatibility(s)
            self.surveys = surveys
        else:
            raise TypeError(
                f"surveys must be a Survey object or an Iterable of Survey objects,"
                f"but surveys is type {type(surveys)}"
            )
        self.surveys = {s.name: s for s in self.surveys}

    def check_compatibility(self, survey):
        """Checks that the compatibility between the survey, the catalog and the generator.

        This should be implemented in subclasses.
        """

    def __iter__(self):
        """Returns iterable which is the object itself."""
        return self

    def _get_psf_from_survey(self, survey: Survey):
        # make PSF and WCS
        psf = []
        for band in survey.available_filters:
            filt = survey.get_filter(band)
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
        return psf

    def __next__(self):
        """Outputs dictionary containing blend output (images and catalogs) in batches.

        Returns:
            output: Dictionary with blend images, isolated object images, blend catalog,
            PSF images and WCS.
        """
        blend_cat = next(self.blend_generator)
        mini_batch_size = np.max([self.batch_size // self.cpus, 1])
        results = BlendResults(self.batch_size, self.max_number, self.stamp_size)
        for s in self.surveys.values():
            pix_stamp_size = int(self.stamp_size / s.pixel_scale.to_value("arcsec"))
            psf = self._get_psf_from_survey(s)
            wcs = make_wcs(s.pixel_scale.to_value("arcsec"), (pix_stamp_size, pix_stamp_size))

            input_args = []
            seedseq_minibatch = self.seedseq.spawn(self.batch_size // mini_batch_size + 1)

            for i in range(0, self.batch_size, mini_batch_size):
                cat = copy.deepcopy(blend_cat[i : i + mini_batch_size])
                input_args.append((cat, psf, wcs, s, seedseq_minibatch[i // mini_batch_size]))

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
            n_bands = len(s.available_filters)
            option1 = (n_bands, pix_stamp_size, pix_stamp_size)
            option2 = (pix_stamp_size, pix_stamp_size, n_bands)
            image_shape = option1 if not self.channels_last else option2

            # organize results.
            blend_images = np.zeros((self.batch_size, *image_shape))
            isolated_images = np.zeros((self.batch_size, self.max_number, *image_shape))
            blend_list = []
            for ii in range(self.batch_size):
                blend_images[ii] = batch_results[ii][0]
                isolated_images[ii] = batch_results[ii][1]
                blend_list.append(batch_results[ii][2])

            results.add_results(s.name, blend_images, isolated_images, blend_list, psf)

        return results

    def render_mini_batch(self, blend_list, psf, wcs, survey, seedseq_minibatch, extra_data=None):
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
            seedseq_minibatch (numpy.random.SeedSequence): Numpy object for generating
                random seeds (for the noise generation).
            extra_data: This field can be used if some data needs to be generated
                before getting to the step where single galaxies are drawn. It should
                have a "shape" of (batch_size, n_blend,...) where n_blend is the number
                of objects in a blend.

        Returns:
            `numpy.ndarray` of blend images and isolated galaxy images, along with
            list of blend catalogs.
        """
        outputs = []
        index = 0

        if extra_data is None:
            extra_data = np.zeros((len(blend_list), np.max([len(blend) for blend in blend_list])))

        # prepare progress bar description
        process_id = get_current_process()
        main_desc = f"Generating blends for {survey.name} survey"
        desc = main_desc if process_id == "main" else f"{main_desc} in process id {process_id}"
        for i, blend in tqdm(enumerate(blend_list), total=len(blend_list), desc=desc):
            # All bands in same survey have same pixel scale, WCS
            pixel_scale = survey.pixel_scale.to_value("arcsec")
            pix_stamp_size = int(self.stamp_size / pixel_scale)

            x_peak, y_peak = get_center_in_pixels(blend, wcs)
            blend.add_column(x_peak)
            blend.add_column(y_peak)

            # add rotation, if requested
            if self.augment_data:
                rng = np.random.default_rng(seedseq_minibatch.generate_state(1))
                theta = rng.uniform(0, 360, size=len(blend))
                blend.add_column(Column(theta), name="btk_rotation")

            n_bands = len(survey.available_filters)
            iso_image_multi = np.zeros((self.max_number, n_bands, pix_stamp_size, pix_stamp_size))
            blend_image_multi = np.zeros((n_bands, pix_stamp_size, pix_stamp_size))
            seedseq_blend = seedseq_minibatch.spawn(n_bands)
            for b, name in enumerate(survey.available_filters):
                filt = survey.get_filter(name)
                single_band_output = self.render_blend(
                    blend, psf[b], filt, survey, seedseq_blend[b], extra_data[i]
                )
                blend_image_multi[b, :, :] = single_band_output[0]
                iso_image_multi[:, b, :, :] = single_band_output[1]

            # transpose if requested.
            dim_order = np.array((0, 1, 2) if not self.channels_last else (1, 2, 0))
            blend_image_multi = blend_image_multi.transpose(dim_order)
            iso_image_multi = iso_image_multi.transpose(0, *(dim_order + 1))

            outputs.append([blend_image_multi, iso_image_multi, blend])
            index += len(blend)
        return outputs

    def render_blend(self, blend_catalog, psf, filt, survey, seedseq_blend, extra_data):
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
            seedseq_blend (numpy.random.SeedSequence): Seed sequence for the noise generation.
            extra_data: Special field of shape (n_blend,?), containing
                additional data for drawing the blend. See render_minibatch
                method for more details.

        Returns:
            Images of blend and isolated galaxies as `numpy.ndarray`.
        """
        sky_level = mean_sky_level(survey, filt).to_value("electron")
        blend_catalog.add_column(
            Column(np.zeros(len(blend_catalog)), name="not_drawn_" + filt.name)
        )
        pix_stamp_size = int(self.stamp_size / survey.pixel_scale.to_value("arcsec"))
        iso_image = np.zeros((self.max_number, pix_stamp_size, pix_stamp_size))
        _blend_image = galsim.Image(np.zeros((pix_stamp_size, pix_stamp_size)))

        for k, entry in enumerate(blend_catalog):
            single_image = self.render_single(entry, filt, psf, survey, extra_data[k])
            if single_image is None:
                iso_image[k] = np.zeros(single_image)
            else:
                iso_image[k] = single_image.array
                _blend_image += single_image

        # add noise.
        if self.add_noise in ("galaxy", "all"):
            if self.verbose:
                print("Galaxy noise added to blend image")
            generator = galsim.random.BaseDeviate(seed=seedseq_blend.generate_state(1))
            galaxy_noise = galsim.PoissonNoise(rng=generator, sky_level=0.0)
            _blend_image.addNoise(galaxy_noise)
        if self.add_noise in ("background", "all"):
            if self.verbose:
                print("Background noise added to blend image")
            generator = galsim.random.BaseDeviate(seed=seedseq_blend.generate_state(1))
            background_noise = galsim.PoissonNoise(rng=generator, sky_level=sky_level)
            noise_image = galsim.Image(np.zeros((pix_stamp_size, pix_stamp_size)))
            noise_image.addNoise(background_noise)
            _blend_image += noise_image

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
                the image is drawn.
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
        for band in survey.available_filters:
            if band + "_ab" not in self.catalog.table.keys():
                raise ValueError(
                    f"The {band} filter of the survey {survey.name} "
                    f"has no associated magnitude in the given catalog."
                )

    def render_single(self, entry, filt, psf, survey, extra_data):
        """Returns the Galsim Image of an isolated galaxy."""
        if self.verbose:
            print("Draw isolated object")

        pix_stamp_size = int(self.stamp_size / survey.pixel_scale.to_value("arcsec"))
        try:
            if self.augment_data:
                entry["pa_bulge"] = (entry["pa_bulge"] + entry["btk_rotation"]) % 360
                entry["pa_disk"] = (entry["pa_disk"] + entry["btk_rotation"]) % 360
            gal = get_catsim_galaxy(entry, filt, survey)
            if self.apply_shear:
                if "g1" in entry.keys() and "g2" in entry.keys():
                    gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
                else:
                    raise KeyError("g1 and g2 not found in blend list.")
            gal_conv = galsim.Convolve(gal, psf)
            gal_conv = gal_conv.shift(entry["ra"], entry["dec"])
            return gal_conv.drawImage(
                nx=pix_stamp_size, ny=pix_stamp_size, scale=survey.pixel_scale.to_value("arcsec")
            )

        except SourceNotVisible:
            if self.verbose:
                print("Source not visible")
            entry["not_drawn_" + filt.name] = 1


class CosmosGenerator(DrawBlendsGenerator):
    """Subclass of DrawBlendsGenerator for drawing galaxies from the COSMOS catalog."""

    def __init__(
        self,
        catalog,
        sampling_function,
        surveys: list,
        batch_size=8,
        stamp_size=24,
        cpus=1,
        verbose=False,
        add_noise="all",
        shifts=None,
        indexes=None,
        channels_last=False,
        save_path=None,
        seed=DEFAULT_SEED,
        gal_type="real",
    ):
        """Initializes the CosmosGenerator class.

        Args:
            catalog (btk.catalog.Catalog): BTK catalog object from which galaxies are taken.
            sampling_function (btk.sampling_function.SamplingFunction): BTK sampling
                function to use.
            surveys (list): List of btk Survey objects defining the observing conditions
            batch_size (int): Number of blends generated per batch
            stamp_size (float): Size of the stamps, in arcseconds
            cpus (int): Number of cpus to use; defines the number of minibatches
            verbose (bool): Indicates whether additionnal information should be printed
            add_noise (str): Indicates if the blends should be generated with noise.
                            "all" indicates that all the noise should be applied,
                            "background" adds only the background noise,
                            "galaxy" only the galaxy noise, and "none" gives noiseless
                            images.
            shifts (list): Contains arbitrary shifts to be applied instead of
                           random shifts. Must be of length batch_size. Must be used
                           with indexes. Used mostly for internal testing purposes.
            indexes (list): Contains the ids of the galaxies to use in the stamp.
                        Must be of length batch_size. Must be used with shifts.
                        Used mostly for internal testing purposes.
            channels_last (bool): Whether to return images as numpy arrays with the channel
                                (band) dimension as the last dimension or before the pixels
                                dimensions (default).
            save_path (str): Path to a directory where results will be saved. If left
                            as None, results will not be saved.
            seed (int): Integer seed for reproducible random noise realizations.
            gal_type (str): string to specify the type of galaxy simulations.
                            Either "real" (default) or "parametric".
        """
        super().__init__(
            catalog=catalog,
            sampling_function=sampling_function,
            surveys=surveys,
            batch_size=batch_size,
            stamp_size=stamp_size,
            cpus=cpus,
            verbose=verbose,
            add_noise=add_noise,
            shifts=shifts,
            indexes=indexes,
            channels_last=channels_last,
            save_path=save_path,
            seed=seed,
        )
        self.gal_type = gal_type

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
            for band in survey.available_filters:
                if f"{survey.name}_{band}" not in self.catalog.table.keys():
                    raise ValueError(
                        f"The {band} filter of the survey {survey.name} "
                        f"has no associated magnitude in the given catalog, "
                        f"and the catalog does not contain a 'ref_mag' column"
                    )

    def render_single(self, entry, filt, psf, survey, extra_data):
        """Returns the Galsim Image of an isolated galaxy."""
        galsim_catalog = self.catalog.get_galsim_catalog()

        # get galaxy flux
        try:
            mag_name = f"{survey.name}_{filt.name}"
            gal_mag = entry[mag_name]
        except KeyError:
            gal_mag = entry["ref_mag"]
        gal_flux = mag2counts(gal_mag, survey, filt).to_value("electron")

        gal = galsim_catalog.makeGalaxy(
            entry["btk_index"], gal_type=self.gal_type, noise_pad_size=0
        ).withFlux(gal_flux)
        if self.augment_data:
            gal.rotate(galsim.Angle(entry["btk_rotation"], unit=galsim.degrees))
        if self.apply_shear:
            if "g1" in entry.keys() and "g2" in entry.keys():
                gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
            else:
                raise KeyError("g1 and g2 not found in blend list.")

        pix_stamp_size = int(self.stamp_size / survey.pixel_scale.to_value("arcsec"))

        # Convolve the galaxy with the PSF
        gal_conv = galsim.Convolve(gal, psf)
        # Apply the shift
        gal_conv = gal_conv.shift(entry["ra"], entry["dec"])

        return gal_conv.drawImage(
            nx=pix_stamp_size, ny=pix_stamp_size, scale=survey.pixel_scale.to_value("arcsec")
        )


class BlendResults(dict):
    """Class which stores the output of DrawBlendGenerator."""

    def __init__(self, batch_size: int, max_n_sources: int, stamp_size: int):
        """Initialize BlendResults class."""
        self.batch_size = batch_size
        self.max_n_sources = max_n_sources
        self.stamp_size = stamp_size  # arcseconds
        self.results = {}

    def _get_pix_stamp_size(self, survey_name: str) -> int:
        return int(self.stamp_size / get_surveys(survey_name).pixel_scale.to_value("arcsec"))

    def _get_wcs(self, survey_name: str):
        pix_stamp_size = self._get_pix_stamp_size(survey_name)
        pixel_scale = get_surveys(survey_name).pixel_scale.to_value("arcsec")
        return make_wcs(pixel_scale, (pix_stamp_size, pix_stamp_size))

    def add_results(
        self,
        survey_name: str,
        blend_images: np.ndarray,
        isolated_images: np.ndarray,
        blend_list: List[Table],
        psfs: List[galsim.GSObject],
    ):
        """Add output of results for a given survey."""
        pix_stamp_size = self._get_pix_stamp_size(survey_name)
        survey = get_surveys(survey_name)
        n_bands = len(survey.available_filters)
        b1, c1, ps11, ps12 = blend_images.shape
        b2, n, c2, ps21, ps22 = isolated_images.shape
        assert b1 == b2 == self.batch_size
        assert c1 == c2 == n_bands
        assert n == self.max_n_sources
        assert ps11 == ps12 == ps21 == ps22 == pix_stamp_size
        self.results[survey_name] = {}
        self.results[survey_name]["blend_images"] = blend_images
        self.results[survey_name]["isolated_images"] = isolated_images
        self.results[survey_name]["blend_list"] = blend_list
        self.results[survey_name]["psfs"] = psfs
        self.results[survey_name]["wcs"] = self._get_wcs(survey_name)
        setattr(self, survey_name, self.results[survey_name])

    def __getitem__(self, key: str):
        """Return key in results dictionary."""
        return self.results[key]

    def __repr__(self):
        string = (
            f"BlendResults(batch_size = {self.batch_size}, "
            f"max_n_sources = {self.max_n_sources}, stamp_size = {self.stamp_size})"
            + ", containing: \n"
        )
        for survey_name in self.results.keys():
            string += survey_name + ": {"
            for key, value in self.results[survey_name].items():
                if isinstance(value, np.ndarray):
                    string += "\n\t" + key + ": np.ndarray, shape " + str(list(value.shape))
                elif isinstance(value, list):
                    item_type = None
                    if len(value) != 0:
                        item_type = value[0].__class__
                    string += (
                        "\n\t" + key + ": list of " + str(item_type) + ", size " + str(len(value))
                    )
                else:
                    string += "\n\t" + key + ": " + str(type(value))
            string += "\n} "
        return string

    def save_results(self, path: str):
        """Save blend results into path."""
        for survey_name in self.results.keys():
            blend_images = self[survey_name]["blend_images"]
            isolated_images = self[survey_name]["isolated_images"]
            blend_list = self[survey_name]["blend_list"]
            psfs = self.results[survey_name]["psfs"]

            if not os.path.exists(os.path.join(path, survey_name)):
                os.makedirs(os.path.join(path, survey_name))

            np.save(os.path.join(path, survey_name, "blend_images.npy"), blend_images)
            np.save(os.path.join(path, survey_name, "isolated_images.npy"), isolated_images)
            with open(os.path.join(path, survey_name, "blend_list.pickle"), "wb") as f:
                pickle.dump(blend_list, f)
            with open(os.path.join(path, survey_name, "psfs.pickle"), "wb") as f:
                pickle.dump(psfs, f)

        # save general info about blend
        with open(os.path.join(path, "blend.json"), "w") as f:
            json.dump(
                {
                    "batch_size": self.batch_size,
                    "max_n_sources": self.max_n_sources,
                    "stamp_size": self.stamp_size,
                },
                f,
            )

    @classmethod
    def load_results(cls, path: str):
        blend_config = json.load(open(os.path.join(path, "blend.json")))
        blend = cls(**blend_config)
        # get folders within the path
        for survey_name in next(os.walk(path))[1]:
            blend_images = np.load(os.path.join(path, survey_name, "blend_images.npy"))
            isolated_images = np.load(os.path.join(path, survey_name, "isolated_images.npy"))
            with open(os.path.join(path, survey_name, "blend_list.pickle"), "rb") as f:
                blend_list = pickle.load(f)
            with open(os.path.join(path, survey_name, "psfs.pickle"), "rb") as f:
                psfs = pickle.load(f)
            blend.add_results(survey_name, blend_images, isolated_images, blend_list, psfs)
        return blend
