"""Module for generating batches of drawn blended images."""
import copy
from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Tuple, Union

import galsim
import numpy as np
from astropy.table import Column, Table
from astropy.wcs import WCS
from galcheat.utilities import mag2counts, mean_sky_level
from tqdm.auto import tqdm

from btk.blend_batch import BlendBatch, MultiResolutionBlendBatch
from btk.blend_generator import BlendGenerator
from btk.catalog import Catalog
from btk.multiprocess import get_current_process, multiprocess
from btk.sampling_functions import SamplingFunction
from btk.survey import Filter, Survey, make_wcs
from btk.utils import DEFAULT_SEED

MAX_SEED_INT = 1_000_000_000


class SourceNotVisible(Exception):
    """Custom exception to indicate that a source has no visible model components."""


def _get_center_in_pixels(blend_table: Table, wcs: WCS):
    """Returns center of objects in blend_catalog in pixel coordinates of postage stamp.

    `blend_catalog` contains `ra, dec` of object center with the postage stamp
    center being 0,0. Coordinates are in pixels where top left corner of postage stamp is (0, 0).

    Args:
        blend_table: Table with entries corresponding to one blend.
        wcs (astropy.wcs.WCS): astropy WCS object corresponding to the image
    Returns:
        `astropy.table.Column`: x and y coordinates of object centroid
    """
    x_peak, y_peak = wcs.world_to_pixel_values(blend_table["ra"] / 3600, blend_table["dec"] / 3600)
    dx_col = Column(x_peak, name="x_peak")
    dy_col = Column(y_peak, name="y_peak")
    return dx_col, dy_col


def get_catsim_galaxy(
    entry: Table,
    filt: Filter,
    survey: Survey,
    no_disk: bool = False,
    no_bulge: bool = False,
    no_agn: bool = False,
):
    """Returns a bulge/disk/agn Galsim galaxy profile based on entry.

    This function returns a composite galsim galaxy profile with bulge, disk and AGN based on the
    parameters in entry, and using observing conditions defined by the survey
    and the filter.

    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        entry: single astropy line containing information on the galaxy
        filt: BTK Filter object
        survey: BTK Survey object
        no_disk: Sets the flux for the disk to zero
        no_bulge: Sets the flux for the bulge to zero
        no_agn: Sets the flux for the AGN to zero

    Returns:
        galsim.GSObject: Galsim galaxy profile
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

    Batch is divided into 'mini-batches' of size `batch_size//njobs` and
    each mini-batch analyzed separately. The results are then combined to output a
    dict with results of entire batch. If the number of njobs is greater than one, then each of
    the mini-batches are run in parallel.
    """

    compatible_catalogs = ("Catalog",)

    def __init__(
        self,
        catalog: Catalog,
        sampling_function: SamplingFunction,
        surveys: Union[List[Survey], Survey],
        batch_size: int = 8,
        stamp_size: float = 24.0,
        njobs: int = 1,
        verbose: bool = False,
        use_bar: bool = False,
        add_noise: str = "all",
        seed: int = DEFAULT_SEED,
        apply_shear: bool = False,
        augment_data: bool = False,
    ):
        """Initializes the DrawBlendsGenerator class.

        Args:
            catalog: BTK catalog object from which galaxies are taken.
            sampling_function: BTK sampling function to use.
            surveys: List of BTK Survey objects or
                single BTK Survey object.
            batch_size: Number of blends generated per batch
            stamp_size: Size of the stamps, in arcseconds
            njobs: Number of njobs to use; defines the number of minibatches
            verbose: Indicates whether additionnal information should be printed
            use_bar: Whether to use progress bar (default: False)
            add_noise: Indicates if the blends should be generated with noise.
                            "all" indicates that all the noise should be applied,
                            "background" adds only the background noise,
                            "galaxy" only the galaxy noise, and "none" gives noiseless
                            images.
            seed: Integer seed for reproducible random noise realizations.
            apply_shear: Whether to apply the shear specified in catalogs to galaxies.
                            If set to True, sampling function must add 'g1', 'g2' columns.
            augment_data: If set to True, augment data by adding a random rotation to every
                            galaxy drawn. Rotation added is stored in the `btk_rotation` column.
        """
        self.blend_generator = BlendGenerator(catalog, sampling_function, batch_size, verbose)
        self.catalog = self.blend_generator.catalog
        self.njobs = njobs
        self.batch_size = self.blend_generator.batch_size
        self.max_number = self.blend_generator.max_number
        self.apply_shear = apply_shear
        self.augment_data = augment_data
        self.stamp_size = stamp_size
        self.use_bar = use_bar
        self._set_surveys(surveys)

        noise_options = {"none", "all", "background", "galaxy"}
        if add_noise not in noise_options:
            raise ValueError(
                f"The options for add_noise are {noise_options}, but you provided {add_noise}"
            )
        self.add_noise = add_noise
        self.verbose = verbose
        self.seedseq = np.random.SeedSequence(seed)

    def _get_pix_stamp_size(self, survey: Survey):
        """Returns the pixel stamp size for a given survey."""
        return int(self.stamp_size / survey.pixel_scale.to_value("arcsec"))

    def _set_surveys(self, surveys):
        """Check if passed in argument `surveys` has correct format."""
        if isinstance(surveys, Survey):
            self.surveys = [surveys]
            self._check_compatibility(surveys)
        elif isinstance(surveys, (tuple, list)):
            for surv in surveys:
                if not isinstance(surv, Survey):
                    raise TypeError(
                        f"surveys must be a Survey object or an Iterable of Survey objects, but "
                        f"Iterable contained object of type {type(surv)}"
                    )
                self._check_compatibility(surv)
            self.surveys = surveys
        else:
            raise TypeError(
                f"surveys must be a Survey object or an Iterable of Survey objects,"
                f"but surveys is type {type(surveys)}"
            )
        self.surveys = {s.name: s for s in self.surveys}

    @abstractmethod
    def _check_compatibility(self, survey: Survey) -> None:
        """Checks that the compatibility between the survey, the catalog and the generator.

        This should be implemented in subclasses.
        """

    def __iter__(self):
        """Returns iterable which is the object itself."""
        return self

    def _get_psf_from_survey(self, survey: Survey) -> List[galsim.GSObject]:
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

    def __next__(self) -> Union[BlendBatch, MultiResolutionBlendBatch]:
        """Outputs dictionary containing blend output in batches.

        Returns:
            `BlendBatch` or `MultiResolutionBlendBatch` object
        """
        blend_cat = next(self.blend_generator)
        mini_batch_size = np.max([self.batch_size // self.njobs, 1])
        blend_batch_list = []
        for surv in self.surveys.values():
            slen = self._get_pix_stamp_size(surv)
            psf = self._get_psf_from_survey(surv)  # psf is the same for all blends in batch.
            wcs = make_wcs(surv.pixel_scale.to_value("arcsec"), (slen, slen))

            input_args = []
            seedseq_minibatch = self.seedseq.spawn(self.batch_size // mini_batch_size + 1)

            for ii in range(0, self.batch_size, mini_batch_size):
                cat = copy.deepcopy(blend_cat[ii : ii + mini_batch_size])
                input_args.append((cat, psf, wcs, surv, seedseq_minibatch[ii // mini_batch_size]))

            # multiprocess and join results
            # ideally, each cpu processes a single mini_batch
            mini_batch_results = multiprocess(
                self._render_mini_batch,
                input_args,
                njobs=self.njobs,
                verbose=self.verbose,
            )

            # join results across mini-batches.
            batch_results = list(chain(*mini_batch_results))

            # organize results.
            n_bands = len(surv.available_filters)
            image_shape = (n_bands, slen, slen)
            blend_images = np.zeros((self.batch_size, *image_shape))
            isolated_images = np.zeros((self.batch_size, self.max_number, *image_shape))
            catalog_list = []
            for ii in range(self.batch_size):
                blend_images[ii] = batch_results[ii][0]
                isolated_images[ii] = batch_results[ii][1]
                catalog_list.append(batch_results[ii][2])

            blend_batch = BlendBatch(
                self.batch_size,
                self.max_number,
                self.stamp_size,
                surv,
                blend_images,
                isolated_images,
                catalog_list,
                psf,
            )

            blend_batch_list.append(blend_batch)

        if len(blend_batch_list) == 1:
            return blend_batch_list[0]

        return MultiResolutionBlendBatch(blend_batch_list)

    def _render_mini_batch(
        self,
        catalog_list: List[Table],
        psf: List[galsim.GSObject],
        wcs: WCS,
        survey: Survey,
        seedseq_minibatch: np.random.SeedSequence,
    ) -> list:
        """Returns isolated and blended images for blend catalogs in catalog_list.

        Function loops over catalog_list and draws blend and isolated images in each
        band. Even though catalog_list was input to the function, we return it since,
        the blend catalogs now include additional columns that flag if an object
        was not drawn and object centers in pixel coordinates.

        Args:
            catalog_list: List of catalogs with entries corresponding to one
                               blend. The size of this list is equal to the
                               mini_batch_size.
            psf: List of Galsim objects containing the PSF
            wcs: astropy WCS object
            survey: Dictionary containing survey information.
            seedseq_minibatch: Numpy object for generating random seeds (for noise generation).

        Returns:
            `numpy.ndarray` of blend images and isolated galaxy images, along with
            list of blend catalogs.
        """
        outputs = []
        index = 0

        # prepare progress bar description
        process_id = get_current_process()
        main_desc = f"Generating blends for {survey.name} survey"
        desc = main_desc if process_id == "main" else f"{main_desc} in process id {process_id}"
        disable = not self.use_bar or process_id != "main"
        for blend in tqdm(catalog_list, total=len(catalog_list), desc=desc, disable=disable):
            # All bands in same survey have same pixel scale, WCS
            slen = self._get_pix_stamp_size(survey)

            x_peak, y_peak = _get_center_in_pixels(blend, wcs)
            blend.add_column(x_peak)
            blend.add_column(y_peak)

            # add rotation, if requested
            if self.augment_data:
                rng = np.random.default_rng(seedseq_minibatch.generate_state(1))
                theta = rng.uniform(0, 360, size=len(blend))
                blend.add_column(Column(theta), name="btk_rotation")
            else:
                blend.add_column(Column(np.zeros(len(blend))), name="btk_rotation")

            n_bands = len(survey.available_filters)
            iso_image_multi = np.zeros((self.max_number, n_bands, slen, slen))
            blend_image_multi = np.zeros((n_bands, slen, slen))
            seedseq_blend = seedseq_minibatch.spawn(n_bands)
            for jj, filter_name in enumerate(survey.available_filters):
                filt = survey.get_filter(filter_name)
                single_band_output = self.render_blend(
                    blend, psf[jj], filt, survey, seedseq_blend[jj]
                )
                blend_image_multi[jj, :, :] = single_band_output[0]
                iso_image_multi[:, jj, :, :] = single_band_output[1]

            outputs.append([blend_image_multi, iso_image_multi, blend])
            index += len(blend)
        return outputs

    def render_blend(
        self,
        blend_catalog: Table,
        psf: galsim.GSObject,
        filt: Filter,
        survey: Survey,
        seedseq_blend: np.random.SeedSequence,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            blend_catalog: Catalog with entries corresponding to one blend.
            psf: Galsim object containing the psf for the given filter
            filt: BTK Filter object
            survey: BTK Survey object
            seedseq_blend: Seed sequence for the noise generation.

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

        for ii, entry in enumerate(blend_catalog):
            single_image = self.render_single(entry, filt, psf, survey)
            if single_image is None:
                iso_image[ii] = np.zeros(single_image)
            else:
                iso_image[ii] = single_image.array
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
    def render_single(self, entry: Table, filt: Filter, psf: galsim.GSObject, survey: Survey):
        """Renders single galaxy in single band in the location given by its entry.

        The image created must be in a stamp of size stamp_size / cutout.pixel_scale. The image
        must be drawn according to information provided by filter, psf, and survey.

        Args:
            entry: Line from astropy describing the galaxy to draw
            filt: BTK Filter object corresponding to the band where
                the image is drawn.
            psf: Galsim object containing the PSF relative to the chosen filter
            survey: BTK Survey object

        Returns:
            galsim.Image object
        """


class CatsimGenerator(DrawBlendsGenerator):
    """Implementation of DrawBlendsGenerator for drawing galaxies from a Catsim-like catalog.

    The code for drawing these galaxies and the default PSF is taken almost directly from
    WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending).
    """

    compatible_catalogs = ("CatsimCatalog",)

    def _check_compatibility(self, survey: Survey) -> None:
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

    def render_single(self, entry: Catalog, filt: Filter, psf: galsim.GSObject, survey: Survey):
        """Returns the Galsim Image of an isolated galaxy."""
        if self.verbose:
            print("Draw isolated object")

        slen = self._get_pix_stamp_size(survey)
        try:
            gal = get_catsim_galaxy(entry, filt, survey)
            gal = gal.rotate(galsim.Angle(entry["btk_rotation"], unit=galsim.degrees))
            if self.apply_shear:
                if "g1" in entry.keys() and "g2" in entry.keys():
                    gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
                else:
                    raise KeyError("g1 and g2 not found in blend list.")
            gal_conv = galsim.Convolve(gal, psf)
            gal_conv = gal_conv.shift(entry["ra"], entry["dec"])
            return gal_conv.drawImage(  # pylint: disable=no-value-for-parameter
                nx=slen,
                ny=slen,
                scale=survey.pixel_scale.to_value("arcsec"),
            )

        except SourceNotVisible:
            if self.verbose:
                print("Source not visible")
            entry["not_drawn_" + filt.name] = 1
            return None


class CosmosGenerator(DrawBlendsGenerator):
    """Subclass of DrawBlendsGenerator for drawing galaxies from the COSMOS catalog."""

    compatible_catalogs = ("CosmosCatalog",)

    def __init__(
        self,
        catalog: Catalog,
        sampling_function: SamplingFunction,
        surveys: List[Survey],
        batch_size: int = 8,
        stamp_size: float = 24.0,
        njobs: int = 1,
        verbose: bool = False,
        add_noise: str = "all",
        seed: int = DEFAULT_SEED,
        use_bar: bool = False,
        apply_shear: bool = False,
        augment_data: bool = False,
        gal_type: str = "real",
    ):
        """Initializes the CosmosGenerator class. See parent class for most attributes.

        Args:
            catalog: See parent class.
            sampling_function: See parent class.
            surveys: See parent class.
            batch_size: See parent class.
            stamp_size: See parent class.
            njobs: See parent class.
            verbose: See parent class.
            add_noise: See parent class.
            seed: See parent class.
            use_bar: See parent class.
            apply_shear: See parent class.
            augment_data: See parent class.
            gal_type: string to specify the type of galaxy simulations.
                            Either "real" (default) or "parametric".
        """
        super().__init__(
            catalog,
            sampling_function,
            surveys,
            batch_size,
            stamp_size,
            njobs,
            verbose,
            use_bar,
            add_noise,
            seed,
            apply_shear,
            augment_data,
        )

        if gal_type not in ("real", "parametric"):
            raise ValueError(
                f"gal_type must be either 'real' or 'parametric', but you provided {gal_type}"
            )
        self.gal_type = gal_type

    def _check_compatibility(self, survey: Survey) -> None:
        if type(self.catalog).__name__ not in self.compatible_catalogs:
            raise ValueError(
                f"The catalog provided is of the wrong type. The types of "
                f"catalogs available for the {type(self).__name__} are {self.compatible_catalogs}"
            )
        for band in survey.available_filters:
            if f"{survey.name}_{band}" not in self.catalog.table.keys():
                raise ValueError(
                    f"The {band} filter of the survey {survey.name} "
                    f"has no associated magnitude in the given catalog."
                )

    def render_single(self, entry: Table, filt: Filter, psf: galsim.GSObject, survey: Survey):
        """Returns the Galsim Image of an isolated galaxy."""
        galsim_catalog = self.catalog.get_galsim_catalog()

        # get galaxy flux
        mag_name = f"{survey.name}_{filt.name}"
        gal_mag = entry[mag_name]
        gal_flux = mag2counts(gal_mag, survey, filt).to_value("electron")

        index = entry["btk_index"]
        gal = galsim_catalog.makeGalaxy(index, gal_type=self.gal_type, noise_pad_size=0)
        gal = gal.withFlux(gal_flux)
        gal = gal.rotate(galsim.Angle(entry["btk_rotation"], unit=galsim.degrees))
        if self.apply_shear:
            if "g1" in entry.keys() and "g2" in entry.keys():
                gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
            else:
                raise KeyError("g1 and g2 not found in blend list.")
        slen = self._get_pix_stamp_size(survey)
        gal_conv = galsim.Convolve(gal, psf)
        gal_conv = gal_conv.shift(entry["ra"], entry["dec"])

        return gal_conv.drawImage(  # pylint: disable=no-value-for-parameter
            nx=slen,
            ny=slen,
            scale=survey.pixel_scale.to_value("arcsec"),
        )
