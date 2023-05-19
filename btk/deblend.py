"""Contains the Measure and MeasureExample classes and its subclasses."""
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import astropy.table
import numpy as np
import sep
from astropy import units
from astropy.coordinates import SkyCoord
from skimage.feature import peak_local_max

from btk.blend_batch import BlendBatch, DeblendedBatch, DeblendedExample
from btk.draw_blends import DrawBlendsGenerator
from btk.multiprocess import multiprocess


class Deblender(ABC):
    """Abstract base class containing the measure class for BTK.

    Each new measure class should be a subclass of Measure.
    """

    @abstractmethod
    def __call__(self, ii: int, blend_batch: BlendBatch) -> DeblendedExample:
        """Implements the call of a measure function on the i-th example.

        Overwrite this function if you perform measurment one image at a time.

        Args:
            blend_batch: Instance of `BlendBatch` class

        Returns:
            Instance of `DeblendedExample` class
        """

    def batch_call(self, blend_batch: BlendBatch, cpus: int = 1) -> DeblendedBatch:
        """Implements the call of a measure function on the entire batch.

        Overwrite this function if you perform measurments on the batch.
        The default fucntionality is to use multiprocessing to speed up
        the iteration over all examples in the batch.

        Args:
            blend_batch: Instance of `BlendBatch` class
            cpus: Number of cpus to paralelize across

        Returns:
            Instance of `DeblendedBatch` class
        """
        args_iter = ((ii, blend_batch) for ii in range(blend_batch.batch_size))
        output = multiprocess(self.__call__, args_iter, cpus=cpus)
        catalog_list = [measured_example.catalog for measured_example in output]
        segmentation, deblended = None, None
        if output[0].segmentation is not None:
            segmentation = np.array([measured_example.segmentation for measured_example in output])
        if output[0].deblended_images is not None:
            deblended = np.array([measured_example.deblended_images for measured_example in output])
        return DeblendedBatch(
            blend_batch.batch_size,
            blend_batch.max_n_sources,
            blend_batch.image_size,
            catalog_list,
            segmentation,
            deblended,
        )

    @classmethod
    def __repr__(cls):
        """Returns the name of the class for bookkeeping."""
        return cls.__name__


class PeakLocalMax(Deblender):
    """This class detects centroids with `skimage.feature.peak_local_max`.

    The function performs detection and deblending of the sources based on the provided
    band index. If use_mean feature is used, then the measurement function is using
    the average of all the bands.
    """

    def __init__(
        self,
        threshold_scale: int = 5,
        min_distance: int = 2,
        use_mean: bool = False,
        use_band: Optional[int] = None,
    ) -> None:
        """Initializes measurement class. Exactly one of 'use_mean' or 'use_band' must be specified.

        Args:
            threshold_scale: Minimum intensity of peaks.
            min_distance: Mimum distance in pixels between two peaks
            use_mean: Flag to use the band average for the measurement
            use_band: Integer index of the band to use for the measurement
        """
        self.min_distance = min_distance
        self.threshold_scale = threshold_scale

        if use_band is None and not use_mean:
            raise ValueError("Either set 'use_mean=True' OR indicate a 'use_band' index")
        if use_band is not None and use_mean:
            raise ValueError("Only one of the parameters 'use_band' and 'use_mean' has to be set")
        self.use_mean = use_mean
        self.use_band = use_band

    def __call__(self, ii: int, blend_batch: BlendBatch) -> DeblendedExample:
        """Performs measurement on the ii-th example from the batch."""
        blend_image = blend_batch.blend_images[ii]
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
        wcs = blend_batch.wcs
        ra, dec = wcs.pixel_to_world_values(x, y)
        ra *= 3600
        dec *= 3600

        # wrap in catalog
        catalog = astropy.table.Table()
        catalog["ra"], catalog["dec"] = ra, dec

        return DeblendedExample(blend_batch.max_n_sources, blend_batch.image_size, catalog)


class SepSingleband(Deblender):
    """Return detection, segmentation and deblending information running SEP on a single band.

    The function performs detection and deblending of the sources based on the provided
    band index. If use_mean feature is used, then the measurement function is using
    the average of all the bands.
    """

    def __init__(
        self,
        sigma_noise: float = 1.5,
        use_mean: bool = False,
        use_band: Optional[int] = None,
    ) -> None:
        """Initializes measurement class. Exactly one of 'use_mean' or 'use_band' must be specified.

        Args:
            sigma_noise: Noise level for sep.
            use_mean: Flag to use the band average for the measurement
            use_band: Integer index of the band to use for the measurement
        """
        if use_band is None and not use_mean:
            raise ValueError("Either set 'use_mean=True' OR indicate a 'use_band' index")
        if use_band is not None and use_mean:
            raise ValueError("Only one of the parameters 'use_band' and 'use_mean' has to be set")
        self.use_mean = use_mean
        self.use_band = use_band
        self.sigma_noise = sigma_noise

    def __call__(self, ii: int, blend_batch: BlendBatch) -> DeblendedExample:
        """Performs measurement on the i-th example from the batch."""
        # get a 1-channel input for sep
        blend_image = blend_batch.blend_images[ii]
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
        for jj in range(n_objects):
            seg_i = segmentation == jj + 1
            segmentation_exp[jj] = seg_i
            deblended_images[jj] = image * seg_i.astype(image.dtype)

        # convert to ra, dec
        wcs = blend_batch._get_wcs()
        ra, dec = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
        ra *= 3600
        dec *= 3600

        # wrap results in astropy table
        cat = astropy.table.Table()
        cat["ra"], cat["dec"] = ra, dec

        return DeblendedExample(
            blend_batch.max_n_sources,
            blend_batch.image_size,
            cat,
            segmentation_exp,
            deblended_images,
        )


class SepMultiband(Deblender):
    """This class returns centers detected with SEP by combining predictions in different bands.

    For each band in the input image we run sep for detection and append new detections to a running
    list of detected coordinates. In order to avoid repeating detections, we run a KD-Tree algorithm
    to calculate the angular distance between each new coordinate and its closest neighbour. Then we
    discard those new coordinates that were closer than matching_threshold to any one of already
    detected coordinates.
    """

    def __init__(self, matching_threshold: float = 1.0, sigma_noise: float = 1.5):
        """Initialize the SepMultiband measurement function.

        Args:
            matching_threshold: Threshold value for match detections that are close
            sigma_noise: Noise level for sep.
        """
        self.matching_threshold = matching_threshold
        self.sigma_noise = sigma_noise

    def __call__(self, ii: int, blend_batch: BlendBatch) -> DeblendedExample:
        """Performs measurement on the ii-th example from the batch."""
        # run source extractor on the first band
        wcs = blend_batch.wcs
        image = blend_batch.blend_images[ii]
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
        return DeblendedExample(blend_batch.max_n_sources, blend_batch.image.size, catalog)


class DeblendGenerator:
    """Generates output of deblender and measurement algorithm."""

    def __init__(
        self,
        deblenders: Union[List[Deblender], Deblender],
        draw_blend_generator: DrawBlendsGenerator,
        cpus: int = 1,
        verbose: bool = False,
    ):
        """Initialize measurement generator.

        Args:
            deblenders: Deblender or a list of Deblender that will be used on the
                            outputs of the draw_blend_generator.
            draw_blend_generator: Instance of subclasses of `DrawBlendsGenerator`.
            cpus: The number of parallel processes to run [Default: 1].
            verbose: Whether to print information about measurement.
        """
        self.deblenders = self._validate_deblenders(deblenders)
        self.measures_names = self._get_unique_deblender_names()
        self.draw_blend_generator = draw_blend_generator
        self.cpus = cpus

        self.batch_size = self.draw_blend_generator.batch_size
        self.verbose = verbose

    def __iter__(self):
        """Return iterator which is the object itself."""
        return self

    def _validate_deblenders(self, deblenders) -> List[Deblender]:
        """Ensure all measure functions are subclasses of `Measure` and correctly instantiated."""
        if not isinstance(deblenders, list):
            deblenders = [deblenders]

        for deblender in deblenders:
            if inspect.isclass(deblender):
                if not issubclass(deblender, Deblender):
                    raise TypeError(f"'{deblender.__name__}' must subclass from Measure")
                raise TypeError(
                    f"'{deblender.__name__}' must be instantiated. Use '{deblender.__name__}()'"
                )
            if not isinstance(deblender, Deblender):
                raise TypeError(
                    f"Got type'{type(deblender)}', but expected an object of a Measure class"
                )
        return deblenders

    def _get_unique_deblender_names(self) -> List[str]:
        """Get list of unique indexed names of each deblender passed in (as necessary)."""
        deblender_names = [str(deblender) for deblender in self.deblenders]
        names_counts = {name: 0 for name in set(deblender_names) if deblender_names.count(name) > 1}
        for ii, name in enumerate(deblender_names):
            if name in names_counts:
                deblender_names[ii] += f"_{names_counts[name]}"
                names_counts[name] += 1
        return deblender_names

    def __next__(self) -> Tuple[BlendBatch, Dict[str, DeblendedBatch]]:
        """Return measurement results on a single batch from the draw_blend_generator.

        Returns:
            draw_blend_generator output from its `__next__` method.
            measurement_results (dict): Dictionary with keys being the name of each
                measure function passed in, and each value its corresponding `MeasuredBatch`.
        """
        blend_output = next(self.draw_blend_generator)
        meas_output = {
            meas_name: meas.batch_call(blend_output)
            for meas_name, meas in zip(self.measures_names, self.deblenders)
        }
        return blend_output, meas_output


available_deblenders = {
    "PeakLocalMax": PeakLocalMax,
    "SepSingleBand": SepSingleband,
    "SepMultiBand": SepMultiband,
}
