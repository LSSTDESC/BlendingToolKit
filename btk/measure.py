"""Contains the Measure and MeasureExample classes and its subclasses."""
import inspect
import json
import os
import pickle
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


class MeasuredExample:
    """Class that validates the results of the measurement for a single image.

    For now, the segmentation and deblended images must correspond only to the
    single `survey_name` survey.

    """

    def __init__(
        self,
        max_n_sources: int,
        stamp_size: int,
        survey_name: str,
        catalog: astropy.table.Table,
        segmentation: np.ndarray = None,
        deblended_images: np.ndarray = None,
    ) -> None:
        """Initializes the measured example."""
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

    def __repr__(self):
        """Return string representation of class."""
        string = (
            f"MeasuredExample(max_n_sources = {self.max_n_sources}, "
            f"stamp_size = {self.stamp_size}, survey_name = {self.survey_name})"
            + ", containing: \n"
        )
        string += "\tcatalog: " + str(type(self.catalog[0]))

        if self.segmentation is not None:
            string += "\n\tsegmentation: np.ndarray, shape " + str(list(self.segmentation.shape))
        else:
            string += "\n\tsegmentation: None"

        if self.deblended_images is not None:
            string += "\n\tdeblended_images: np.ndarray, shape " + str(list(self.deblended_images.shape))
        else:
            string += "\n\tdeblended_images: None"
        return string

class MeasuredBatch:
    """Class that validates the results of the measurement for a batch of image."""

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
        """Initializes the measured example."""
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
        return catalog_list

    def _validate_segmentation(self, segmentation):
        if segmentation is not None:
            assert segmentation.shape == (
                self.batch_size,
                self.max_n_sources,
                self.image_size,
                self.image_size,
            )
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

    def __repr__(self):
        """Return string representation of class."""
        string = (
            f"MeasuredBatch(batch_size = {self.batch_size}, "
            f"max_n_sources = {self.max_n_sources}, stamp_size = {self.stamp_size}, survey_name = {self.survey_name})"
            + ", containing: \n"
        )
        string += "\tcatalog: list of " + str(type(self.catalog[0])) + ", size " + str(len(self.catalog))

        if self.segmentation is not None:
            string += "\n\tsegmentation: np.ndarray, shape " + str(list(self.segmentation.shape))
        else:
            string += "\n\tsegmentation: None"

        if self.deblended_images is not None:
            string += "\n\tdeblended_images: np.ndarray, shape " + str(list(self.deblended_images.shape))
        else:
            string += "\n\tdeblended_images: None"
        return string

    def save_batch(self, path: str, batch_number: int) -> None:
        """Save batch of measure results to disk."""
        save_dir = os.path.join(path, str(batch_number), self.survey_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            np.save(os.path.join(save_dir, "segmentation"), self.segmentation)
            np.save(os.path.join(save_dir, "deblended_images"), self.deblended_images)
            with open(os.path.join(save_dir, "catalog.pickle"), "wb") as f:
                pickle.dump(self.catalog, f)

        # save general info about class
        with open(os.path.join(path, "meas.json"), "w") as f:
            json.dump(
                {
                    "batch_size": self.batch_size,
                    "max_n_sources": self.max_n_sources,
                    "stamp_size": self.stamp_size,
                    "survey_name": self.survey_name,
                },
                f,
            )

    @classmethod
    def load_batch(cls, path: str, survey_name: str, batch_number: int):
        """Load batch of measure results from disk."""
        load_dir = os.path.join(path, str(batch_number), survey_name)
        with open(os.path.join(path, "meas.json"), "r") as f:
            meas_config = json.load(f)
        assert meas_config["survey_name"] == survey_name

        with open(os.path.join(load_dir, "catalog.pickle"), "rb") as f:
            catalog = pickle.load(f)
        segmentation = np.load(os.path.join(load_dir, "segmentation.npy"))
        deblended_images = np.load(os.path.join(load_dir, "deblended_images.npy"))
        return cls(
            catalog=catalog,
            segmentation=segmentation,
            deblended_images=deblended_images,
            **meas_config,
        )


class Measure(ABC):
    """Abstract base class containing the measure class for BTK.

    Each new measure class should be a subclass of Measure.
    """

    @abstractmethod
    def __call__(self, i: int, blend_batch: BlendBatch) -> MeasuredExample:
        """Implements the call of a measure function on the i-th example.

        Overwrite this function if you perform measurment one image at a time.

        Args:
            blend_batch: Instance of `BlendBatch` class

        Returns:
            Instance of `MeasuredExample` class
        """

    def batch_call(self, blend_batch: BlendBatch, cpus: int = 1) -> MeasuredBatch:
        """Implements the call of a measure function on the entire batch.

        Overwrite this function if you perform measurments on the batch.
        The default fucntionality is to use multiprocessing to speed up
        the iteration over all examples in the batch.

        Args:
            blend_batch: Instance of `BlendBatch` class
            cpus: Number of cpus to paralelize across

        Returns:
            Instance of `MeasuredBatch` class
        """
        args_iter = ((i, blend_batch) for i in range(blend_batch.batch_size))
        output = multiprocess(
            self.__call__,
            args_iter,
            cpus=cpus,
        )
        catalog_list = [measured_example.catalog for measured_example in output]
        segmentation, deblended = None, None
        if output[0].segmentation is not None:
            segmentation = np.array([measured_example.segmentation for measured_example in output])
        if output[0].deblended_images is not None:
            deblended = np.array([measured_example.deblended_images for measured_example in output])
        return MeasuredBatch(
            max_n_sources=blend_batch.max_n_sources,
            stamp_size=blend_batch.stamp_size,
            batch_size=blend_batch.batch_size,
            survey_name=output[0].survey_name,
            catalog_list=catalog_list,
            segmentation=segmentation,
            deblended_images=deblended,
        )

    @classmethod
    def __repr__(cls):
        """Returns the name of the class for bookkeeping."""
        return cls.__name__


class PeakLocalMax(Measure):
    """This class detects centroids with `skimage.feature.peak_local_max`.

    The function performs detection and deblending of the sources based on the provided
    band index. If use_mean feature is used, then the measurement function is using
    the average of all the bands.
    """

    def __init__(
        self,
        threshold_scale: int = 5,
        min_distance: int = 2,
        survey_name: str = None,
        use_band: int = None,
        use_mean: bool = False,
    ):
        """Initializes measurement class. Exactly one of 'use_mean' or 'use_band' must be specified.

        Args:
            threshold_scale: Minimum intensity of peaks.
            min_distance: Mimum distance in pixels between two peaks
            survey_name: Name of the survey to measure on
            use_mean: Flag to use the band average for the measurement
            use_band: Integer index of the band to use for the measurement
        """
        self.min_distance = min_distance
        self.threshold_scale = threshold_scale
        if survey_name is None:
            raise ValueError("'survey_name' must be specified for this measurement")
        self.survey_name = survey_name
        if use_band is None and not use_mean:
            raise ValueError("Either set 'use_mean=True' OR indicate a 'use_band' index")
        if use_band is not None and use_mean:
            raise ValueError("Only one of the parameters 'use_band' and 'use_mean' has to be set")
        self.use_mean = use_mean
        self.use_band = use_band

    def __call__(self, i: int, blend_batch: BlendBatch) -> MeasuredExample:
        """Performs measurement on the i-th example from the batch."""
        blend_image = blend_batch[self.survey_name]["blend_images"][i]
        if self.use_mean:
            image = np.mean(blend_image, axis=0)
        else:
            image = blend_image[self.use_band]

        # compute threshold value
        threshold = self.threshold_scale * np.std(image)

        # calculate coordinates
        coordinates = peak_local_max(image, min_distance=self.min_distance, threshold_abs=threshold)
        x, y = coordinates[:, 1], coordinates[:, 0]

        # convert coordinates to ra, dec
        wcs = blend_batch[self.survey_name]["wcs"]
        ra, dec = wcs.pixel_to_world_values(x, y)
        ra *= 3600
        dec *= 3600

        # wrap in catalog
        catalog = astropy.table.Table()
        catalog["ra"], catalog["dec"] = ra, dec

        return MeasuredExample(
            max_n_sources=blend_batch.max_n_sources,
            stamp_size=blend_batch.stamp_size,
            survey_name=self.survey_name,
            catalog=catalog,
        )


class SepSingleband(Measure):
    """Return detection, segmentation and deblending information running SEP on a single band.

    The function performs detection and deblending of the sources based on the provided
    band index. If use_mean feature is used, then the measurement function is using
    the average of all the bands.
    """

    def __init__(self, sigma_noise=1.5, use_mean=False, use_band=None, survey_name=None):
        """Initializes measurement class. Exactly one of 'use_mean' or 'use_band' must be specified.

        Args:
            survey_name: Name of the survey to measure on
            use_mean: Flag to use the band average for the measurement
            use_band: Integer index of the band to use for the measurement
            sigma_noise: Noise level for sep.
        """
        if survey_name is None:
            raise ValueError("'survey_name' must be specified for this measurement")
        self.survey_name = survey_name
        if use_band is None and not use_mean:
            raise ValueError("Either set 'use_mean=True' OR indicate a 'use_band' index")
        if use_band is not None and use_mean:
            raise ValueError("Only one of the parameters 'use_band' and 'use_mean' has to be set")
        self.use_mean = use_mean
        self.use_band = use_band
        self.sigma_noise = sigma_noise

    def __call__(self, i: int, blend_batch: BlendBatch):
        """Performs measurement on the i-th example from the batch."""
        # get a 1-channel input for sep
        blend_image = blend_batch[self.survey_name]["blend_images"][i]
        if self.use_mean:
            image = np.mean(blend_image, axis=0)
        else:
            image = blend_image[self.use_band]
        assert image.ndim == 2

        # run source extractor
        bkg = sep.Background(image)
        catalog, segmentation = sep.extract(
            image, self.sigma_noise, err=bkg.globalrms, segmentation_map=True
        )

        n_objects = len(catalog)
        segmentation_exp = np.zeros((blend_batch.max_n_sources, *image.shape), dtype=bool)
        deblended_images = np.zeros((blend_batch.max_n_sources, *image.shape), dtype=image.dtype)
        for i in range(n_objects):
            seg_i = segmentation == i + 1
            segmentation_exp[i] = seg_i
            deblended_images[i] = image * seg_i.astype(image.dtype)

        # convert to ra, dec
        wcs = blend_batch[self.survey_name]["wcs"]
        ra, dec = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
        ra *= 3600
        dec *= 3600

        # wrap results in astropy table
        t = astropy.table.Table()
        t["ra"], t["dec"] = ra, dec

        return MeasuredExample(
            max_n_sources=blend_batch.max_n_sources,
            stamp_size=blend_batch.stamp_size,
            survey_name=self.survey_name,
            catalog=t,
            segmentation=segmentation_exp,
            deblended_images=deblended_images,
        )


class SepMultiband(Measure):
    """This class returns centers detected with SEP by combining predictions in different bands.

    For each band in the input image we run sep for detection and append new detections to a running
    list of detected coordinates. In order to avoid repeating detections, we run a KD-Tree algorithm
    to calculate the angular distance between each new coordinate and its closest neighbour. Then we
    discard those new coordinates that were closer than matching_threshold to any one of already
    detected coordinates.
    """

    def __init__(self, matching_threshold=1.0, sigma_noise=1.5, survey_name=None):
        """Initialize the SepMultiband measurement function.

        Args:
            survey_name: Name of the survey to measure on.
            sigma_noise: Noise level for sep.
            matching_threshold: Threshold value for match detections that are close
        """
        if survey_name is None:
            raise ValueError("'survey_name' must be specified for this measurement")
        self.survey_name = survey_name
        self.matching_threshold = matching_threshold
        self.sigma_noise = sigma_noise

    def __call__(self, i, blend_batch):
        """Performs measurement on the i-th example from the batch."""
        # run source extractor on the first band
        wcs = blend_batch[self.survey_name]["wcs"]
        image = blend_batch[self.survey_name]["blend_images"][i]
        bkg = sep.Background(image[0])
        catalog = sep.extract(image[0], self.sigma_noise, err=bkg.globalrms, segmentation_map=False)
        ra_coordinates, dec_coordinates = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
        ra_coordinates *= 3600
        dec_coordinates *= 3600

        # iterate over remaining bands and match predictions using KdTree
        for band in range(1, image.shape[0]):
            # run source extractor
            band_image = image[band]
            bkg = sep.Background(band_image)
            catalog = sep.extract(
                band_image, self.sigma_noise, err=bkg.globalrms, segmentation_map=False
            )

            # convert predictions to arcseconds
            ra_detections, dec_detections = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
            ra_detections *= 3600
            dec_detections *= 3600

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
        catalog = astropy.table.Table()
        catalog["ra"] = ra_coordinates
        catalog["dec"] = dec_coordinates
        return MeasuredExample(
            max_n_sources=blend_batch.max_n_sources,
            stamp_size=blend_batch.stamp_size,
            survey_name=self.survey_name,
            catalog=catalog,
        )


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
        meas_output = {
            meas_name: meas.batch_call(blend_output)
            for meas_name, meas in zip(self.measures_names, self.measures)
        }
        return blend_output, meas_output


available_measure_functions = {
    "PeakLocalMax": PeakLocalMax,
    "SepSingleBand": SepSingleband,
    "SepMultiBand": SepMultiband,
}
