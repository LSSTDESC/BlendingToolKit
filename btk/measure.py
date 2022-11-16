import inspect
import os
from abc import ABC, abstractmethod
from typing import List, Union

import astropy.table
import numpy as np
import sep
from astropy import units
from astropy.coordinates import SkyCoord
from skimage.feature import peak_local_max

from btk.draw_blends import BlendBatch
from btk.multiprocess import multiprocess
from btk.survey import get_surveys
from btk.utils import add_pixel_columns, reverse_list_dictionary

class MeasuredExample:
    """Class that validates the results of the measurement for a single image"""

    def __init__(
        self,
        max_n_sources: int,
        stamp_size: int,
        survey_name: str,
        catalog: astropy.table.Table,
        segmentation: np.ndarray = None,
        deblended_images: np.ndarray = None,
    ) -> None:
        """Initializes the measured example"""

        self.max_n_sources = max_n_sources
        self.stamp_size = stamp_size
        self.survey_name = survey_name
        self.image_size = self.stamp_size / get_surveys(survey_name).pixel_scale.to_value("arcsec")

        self.catalog = self._validate_catalog(catalog)
        self.segmentation = self._validate_segmentation(segmentation) 
        self.deblended_images = self._validate_deblended_images(deblended_images) 

    def _validate_catalog(self, catalog: astropy.table.Table):

        if not ("ra" in catalog.colnames and "dec" in catalog.colnames):
            raise ValueError(
                "The output catalog of at least one of your measurement functions does"
                "not contain the mandatory 'ra' and 'dec' columns"
            )
        
        if (catalog["ra"] < 0).any() or (catalog["ra"] > self.stamp_size).any():
            raise ValueError(
                f"All 'ra' detections must be in range [0, {self.stamp_size}]"
            )
        
        if (catalog["dec"] < 0).any() or (catalog["dec"] > self.stamp_size).any():
            raise ValueError(
                f"All 'dec' detections must be in range [0, {self.stamp_size}]"
            )
        return catalog

    def _validate_segmentation(self, segmentation):
        if segmentation is not None:
            assert segmentation.shape == (self.max_n_sources, self.image_size, self.image_size)
            assert segmentation.min() >= 0 and segmentation.max() <= 1
        return segmentation

    def _validate_deblended_images(self, deblended_images):
        if deblended_images is not None:
            assert deblended_images.shape == (
                self.max_n_sources,
                self.image_size,
                self.image_size,
            )
        return deblended_images

class MeasuredBatch:
    """Class that validates the results of the measurement for a batch of image"""

    def __init__(
        self,
        max_n_sources: int,
        stamp_size: int,
        batch_size: int,
        survey_name: str,
        catalog_list: List[astropy.table.Table],
        segmentation: np.ndarray = None,
        deblended_images: np.ndarray = None,
    ) -> None:
        """Initializes the measured example"""

        self.max_n_sources = max_n_sources
        self.stamp_size = stamp_size
        self.batch_size = batch_size
        self.survey_name = survey_name
        self.image_size = self.stamp_size / get_surveys(survey_name).pixel_scale.to_value("arcsec")

        self.catalog = self._validate_catalog(catalog_list)
        self.segmentation = self._validate_segmentation(segmentation) 
        self.deblended_images = self._validate_deblended_images(deblended_images) 

    def _validate_catalog(self, catalog_list: List[astropy.table.Table]):

        if not isinstance(catalog_list, list):
            raise TypeError(
                "Catalog must be a list of 'astropy.table.Table' for each image in the batch"
            )
        assert len(catalog_list) == self.batch_size
        for catalog in catalog_list:
            if not ("ra" in catalog.colnames and "dec" in catalog.colnames):
                raise ValueError(
                    "The output catalog of at least one of your measurement functions does"
                    "not contain the mandatory 'ra' and 'dec' columns"
                )
            
            if (catalog["ra"] < 0).any() or (catalog["ra"] > self.stamp_size).any():
                raise ValueError(
                    f"All 'ra' detections must be in range [0, {self.stamp_size}]"
                )
            
            if (catalog["dec"] < 0).any() or (catalog["dec"] > self.stamp_size).any():
                raise ValueError(
                    f"All 'dec' detections must be in range [0, {self.stamp_size}]"
                )
        return catalog_list

    def _validate_segmentation(self, segmentation):
        if segmentation is not None:
            assert segmentation.shape == (self.batch_size, self.max_n_sources, self.image_size, self.image_size)
            assert segmentation.min() >= 0 and segmentation.max() <= 1
        return segmentation

    def _validate_deblended_images(self, deblended_images):
        if deblended_images is not None:
            assert deblended_images.shape == (
                self.batch_size, 
                self.max_n_sources,
                self.image_size,
                self.image_size,
            )
        return deblended_images


class Measure(ABC):
    """Abstract base class containing the measure class for BTK.

    Each new measure class should be a subclass of Measure.
    """

    def __call__(self, i,  blend_results: BlendBatch) -> MeasuredExample:
        """Implements the call of a measure function on the i-th example.

        Overwrite this function if you perform measurment one image at a time.

        Args:
            blend_results: Instance of `BlendResults` class

        Returns:
            Instance of `MeasuredExample` class
        """
        raise NotImplementedError("Each measure class must implement its own `__call__` function")
    
    def __batch_call__(self, blend_results: BlendBatch) -> MeasuredBatch:
        """Implements the call of a measure function on the entire batch.
        
        Overwrite this function if you perform measurments on the batch.
        The default fucntionality is to use multiprocessing to speed up
        the iteration over all examples in the batch.

        Args:
            blend_results: Instance of `BlendResults` class

        Returns:
            Instance of `MeasuredBatch` class
        """
        args_iter = ((i, blend_results) for i in range(blend_results.batch_size))
        output = multiprocess(
            self.__call__,
            args_iter,
            cpus=self.cpus,
        )
        # TODO convert output to MeasuredBatch
        mb = MeasuredBatch()
        return MeasuredBatch
        

    @classmethod
    def __repr__(cls):
        """Returns the name of the class for bookkeeping."""
        return cls.__name__


class PeakLocalMax(Measure):
    """This measure function returns centroids detected with `skimage.feature.peak_local_max`.

    For each potentially multi-band image, an average over the bands is taken before measurement.


    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.

    Returns:
        dict containing catalog with entries corresponding to measured peaks.
    """

    def __init__(self, threshold_scale=5, min_distance=2):
        self.min_distance = min_distance
        self.threshold_scale = threshold_scale

    def __call__(self, i, blend_results, survey_names):

        if not isinstance(survey_names, list):
            survey_names = [survey_names]

        for survey_name in survey_names:
            for image in blend_results[survey_name]["blend_images"]:
                # take a band average
                avg_image = np.mean(image, axis=1)

                # compute threshold value
                threshold = self.threshold_scale * np.std(avg_image)

                # calculate coordinates
                coordinates = peak_local_max(
                    avg_image, min_distance=self.min_distance, threshold_abs=threshold
                )
                x, y = coordinates[:, 1], coordinates[:, 0]
                # convert coordinates to ra, dec
                wcs = blend_results[survey_name]["wcs"]
                ra, dec = wcs.pixel_to_world_values(x, y)
                ra *= 3600
                dec *= 3600

                # wrap in catalog
                catalog = astropy.table.Table()
                catalog["ra"] = ra
                catalog["dec"] = dec
                measure_results.add_results(survey_name, catalog)

        return measure_results     

       


class SepSingleband(Measure):
    """TODO: Docstring.

    Return detection, segmentation and deblending information running SEP on a single band.
    The function performs detection and deblending of the sources based on the provided
    band index. If use_mean feature is used, then the measurement function is using
    the average of all the bands.

    NOTE: If this function is used with the multiresolution feature,
    measurements will be carried on the first survey, and deblended images
    or segmentations will not be returned.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.
        meas_band_num (int): Indicates the index of band to use fo the measurement
        use_mean (bool): If True, then algorithm uses the average of all the bands
        sigma_noise (float): Sigma threshold for detection against noise.

    Returns:
        dict with the centers of sources detected by SEP detection algorithm.
    """

    def __init__(self, use_mean=False, meas_band_num=3, sigma_noise=1.5, channels_last=False):
        """TODO: Docstring."""
        self.use_mean = use_mean
        self.meas_band_num = meas_band_num
        self.sigma_noise = sigma_noise
        self.channels_last = channels_last
        self.channels_index = 0 if not channels_last else -1

    def __call__(self, image, wcs):
        """TODO: Docstring."""
        # get a 1-channel input for sep
        if self.use_mean:
            band_image = np.mean(image, axis=self.channels_index)
        else:
            band_image = (
                image[self.meas_band_num]
                if self.channels_index == 0
                else image[:, :, self.meas_band_num]
            )

        # run source extractor
        stamp_size = band_image.shape[0]
        bkg = sep.Background(band_image)
        catalog, segmentation = sep.extract(
            band_image, self.sigma_noise, err=bkg.globalrms, segmentation_map=True
        )

        # reshape segmentation map
        n_objects = len(catalog)
        segmentation_exp = np.zeros((n_objects, stamp_size, stamp_size), dtype=bool)
        deblended_images = np.zeros((n_objects, *image.shape), dtype=image.dtype)
        for i in range(n_objects):
            seg_i = segmentation == i + 1
            segmentation_exp[i] = seg_i
            seg_i_reshaped = np.zeros((np.min(image.shape), stamp_size, stamp_size))
            for j in range(np.min(image.shape)):
                seg_i_reshaped[j] = seg_i
            seg_i_reshaped = np.moveaxis(seg_i_reshaped, 0, np.argmin(image.shape))
            deblended_images[i] = image * seg_i_reshaped

        # wrap results in astropy table
        t = astropy.table.Table()
        t["ra"], t["dec"] = self.pixel_coordinates_to_arcsec(catalog["x"], catalog["y"], wcs)
        return {
            "catalog": t,
            "segmentation": segmentation_exp,
            "deblended_images": deblended_images,
        }


class SepMultiband(Measure):
    """This class.

    Returns centers detected with source extractor by combining predictions in different bands.
    For each band in the input image we run sep for detection and append new detections to a running
    list of detected coordinates. In order to avoid repeating detections, we run a KD-Tree algorithm
    to calculate the angular distance between each new coordinate and its closest neighbour. Then we
    discard those new coordinates that were closer than matching_threshold to any one of already
    detected coordinates.

    NOTE: If this function is used with the multiresolution feature,
    measurements will be carried on the first survey.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.
        sigma_noise (float): Sigma threshold for detection against noise.
        matching_threshold (float): Match centers of objects that are closer than
            this threshold to a single prediction (in arseconds).

    Returns:
        dict containing catalog with entries corresponding to measured peaks.
    """

    def __init__(self, matching_threshold=1.0, sigma_noise=1.5, channels_last=False):
        """Initialize the SepMultiband measurement function."""
        self.matching_threshold = matching_threshold
        self.sigma_noise = sigma_noise
        self.channels_last = channels_last
        self.channels_index = 0 if not channels_last else -1

    def __call__(self, image: np.ndarray, wcs):
        """TODO: Docstring."""
        # run source extractor on the first band
        band_image = image[:, :, 0] if self.channels_last else image[0]
        bkg = sep.Background(band_image)
        catalog = sep.extract(
            band_image, self.sigma_noise, err=bkg.globalrms, segmentation_map=False
        )
        ra_coordinates, dec_coordinates = self.pixel_coordinates_to_arcsec(
            catalog["x"], catalog["y"], wcs
        )

        # iterate over remaining bands and match predictions using KdTree
        for band in range(1, image.shape[self.channels_index]):
            # run source extractor
            band_image = image[band] if self.channels_index == 0 else image[:, :, band]
            bkg = sep.Background(band_image)
            catalog = sep.extract(
                band_image, self.sigma_noise, err=bkg.globalrms, segmentation_map=False
            )

            # convert predictions to arcseconds
            ra_detections, dec_detections = self.pixel_coordinates_to_arcsec(
                catalog["x"], catalog["y"], wcs
            )

            # convert to sky coordinates
            c1 = SkyCoord(ra=ra_detections * units.arcsec, dec=dec_detections * units.arcsec)
            c2 = SkyCoord(ra=ra_coordinates * units.arcsec, dec=dec_coordinates * units.arcsec)

            # merge new detections with the running list of coordinates
            if len(c1) > 0 and len(c2) > 0:
                # runs KD-tree to get distances to the closest neighbours
                _, distance2d, _ = c1.match_to_catalog_sky(c2)
                distance2d = distance2d.arcsec

                # add new predictions, masking those that are closer than threshold
                ra_coordinates = np.concatenate(
                    [ra_coordinates, ra_detections[distance2d > self.matching_threshold]]
                )
                dec_coordinates = np.concatenate(
                    [dec_coordinates, dec_detections[distance2d > self.matching_threshold]]
                )
            else:
                ra_coordinates = np.concatenate([ra_coordinates, ra_detections])
                dec_coordinates = np.concatenate([dec_coordinates, dec_detections])

        # Wrap in the astropy table
        t = astropy.table.Table()
        t["ra"] = ra_coordinates
        t["dec"] = dec_coordinates
        return {"catalog": t}


class MeasureGenerator:
    """Generates output of deblender and measurement algorithm."""

    def __init__(
        self,
        measures: Union[List[Measure], Measure],
        draw_blend_generator,
        cpus=1,
        verbose=False,
        save_path=None,
    ):
        """Initialize measurement generator.

        Args:
            measures: Measure or a list of Measures that will be performed on the
                outputs of the draw_blend_generator.
            draw_blend_generator: Generator that outputs dict with blended images,
                                isolated images, blend catalog, wcs info, and psf.
            cpus: The number of parallel processes to run [Default: 1].
            verbose (bool): Whether to print information about measurement.
            save_path (str): Path to a directory where results will be saved. If left
                            as None, results will not be saved.
        """
        self.measures = self._validate_measure_functions(measures)
        self.measures_names = self._get_unique_measure_names()
        self.draw_blend_generator = draw_blend_generator
        self.cpus = cpus

        self.batch_size = self.draw_blend_generator.batch_size
        self.verbose = verbose
        self.save_path = save_path

    def __iter__(self):
        """Return iterator which is the object itself."""
        return self

    def _validate_measure_functions(self, measures) -> List[Measure]:
        if not isinstance(measures, list):
            measures = [measures]

        for meas in measures:
            if inspect.isclass(meas):
                if not issubclass(meas, Measure):
                    raise TypeError(f"'{meas.__name__}' must subclass from Measure")
                else:
                    raise TypeError(
                        f"'{meas.__name__}' must be instantiated. Use '{meas.__name__}()' instead"
                    )
            elif not isinstance(meas, Measure):
                raise TypeError(
                    f"Got type'{type(meas)}', but expected an object of a Measure class"
                )
        return measures

    def _get_unique_measure_names(self) -> List[str]:
        """Get list of unique indexed names of each measure function passed in (as necessary)."""
        measures_names = [str(meas) for meas in self.measures]
        names_counts = {name: 0 for name in set(measures_names) if measures_names.count(name) > 1}
        for ii, name in enumerate(measures_names):
            if name in names_counts:
                measures_names[ii] += f"_{names_counts[name]}"
                names_counts[name] += 1
        return measures_names

    def _run_batch(self, batch, index):
        """Perform measurements on a single blend."""
        output = []
        for meas in self.measures:
            out = meas(batch["blend_images"][index], batch["wcs"][index])
            out = {k: out.get(k, None) for k in ["deblended_images", "catalog", "segmentation"]}
            output.append(out)
        return output

    def __next__(self):
        """Return measurement results on a single batch from the draw_blend_generator.

        Returns:
            draw_blend_generator output from its `__next__` method.
            measurement_results (dict): Dictionary with keys being the name of each
                `measure_function` passed in. Each value is a dictionary containing keys
                `catalog`, `deblended_images`, and `segmentation` storing the values returned by
                the corresponding measure_function` for one batch.
        """
        blend_output = next(self.draw_blend_generator)
        catalog = {}
        segmentation = {}
        deblended_images = {}
        for name in self.measures_names:
            catalog[name] = []
            segmentation[name] = []
            deblended_images[name] = []

        args_iter = ((blend_output, i) for i in range(self.batch_size))
        measure_output = multiprocess(
            self._run_batch,
            args_iter,
            cpus=self.cpus,
            verbose=self.verbose,
        )

        # organize outputs from multiprocessing
        for ii in range(len(measure_output)):
            for jj, meas_name in enumerate(self.measures):
                catalog[meas_name].append(measure_output[ii][jj].get("catalog", None))
                segmentation[meas_name].append(measure_output[ii][jj].get("segmentation", None))
                deblended_images[meas_name].append(
                    measure_output[ii][jj].get("deblended_images", None)
                )

                if self.is_multiresolution:
                    self._cleanup_multiresolution_output(
                        catalog, segmentation, deblended_images, blend_output, meas_name
                    )

                else:
                    catalog[meas_name] = add_pixel_columns(catalog[meas_name], blend_output["wcs"])

                self._save_batch(catalog, segmentation, deblended_images, meas_name)

        measure_results = {
            "catalog": catalog,
            "segmentation": segmentation,
            "deblended_images": deblended_images,
        }
        return blend_output, measure_results

    def _cleanup_multiresolution_output(
        self,
        catalog: dict,
        segmentation: dict,
        deblended_images: dict,
        blend_output: dict,
        meas_name: str,
    ):
        # If multiresolution, we reverse the order between the survey name and
        # the index of the blend
        # TODO Fix this -- this seems very weird and broken...
        # We duplicate the catalog for each survey to get the pixel coordinates
        catalogs_temp = {}
        for surv in self.survey_names:
            catalogs_temp[surv] = add_pixel_columns(catalog[meas_name], blend_output["wcs"][surv])
        catalog[meas_name] = catalogs_temp

        segmentation[meas_name] = reverse_list_dictionary(
            segmentation[meas_name], self.survey_names
        )
        deblended_images[meas_name] = reverse_list_dictionary(
            deblended_images[meas_name], self.survey_names
        )

    def _save_batch(
        self, catalog: dict, segmentation: dict, deblended_images: dict, meas_name: str
    ):
        if self.save_path is not None:
            if not os.path.exists(os.path.join(self.save_path, meas_name)):
                os.mkdir(os.path.join(self.save_path, meas_name))

            if segmentation[meas_name] is not None:
                np.save(
                    os.path.join(self.save_path, meas_name, "segmentation"),
                    np.array(segmentation[meas_name], dtype=object),
                )
            if deblended_images[meas_name] is not None:
                np.save(
                    os.path.join(self.save_path, meas_name, "deblended_images"),
                    np.array(deblended_images[meas_name], dtype=object),
                )
            for j, cat in enumerate(catalog[meas_name]):
                cat.write(
                    os.path.join(self.save_path, meas_name, f"detection_catalog_{j}"),
                    format="ascii",
                    overwrite=True,
                )


available_measure_functions = {
    "PeakLocalMax": PeakLocalMax,
    "SepSingleBand": SepSingleband,
    "SepMultiBand": SepMultiband,
}
