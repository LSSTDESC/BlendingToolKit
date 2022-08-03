"""File containing measurement infrastructure for the BlendingToolKit.

Contains examples of functions that can be used to apply a measurement algorithm to the blends
simulated by BTK. Every measurement function should have the following skeleton:

::

    def measure_function(batch, idx, **kwargs):
        # do some measurements on the images contained in batch.
        return output


where `batch` is the output from the `DrawBlendsGenerator` object (see its `__next__` method) and
`idx` is the index corresponding to which image in the batch to measure. The additional keyword
arguments `**kwargs` can be passed via the `measure_kwargs` dictionary argument in the
`MeasureGenerator` initialize which are shared among all the measurement functions.

It should return a dictionary containing a subset of the following keys/values (note the key
catalog` is mandatory):

* catalog (astropy.table.Table): An astropy table containing measurement information. The
  `len` of the table should be `n_objects`. If your
  DrawBlendsGenerator uses a single survey, the following
  column names are required:

  * x_peak: horizontal centroid position in pixels.
  * y_peak: vertical centroid position in pixels.

  For multiple surveys (multi-resolution), we instead require:

  * ra: object centroid right ascension in arcseconds,
    following the convention from the `wcs` object included in
    the input batch.
  * dec: vertical centroid position in arcseconds,
    following the convention from the `wcs` object included in
    the input batch.

* deblended_image (np.ndarray): Array of deblended isolated images with shape:
  `(n_objects, n_bands, stamp_size, stamp_size)` or
  `(n_objects, stamp_size, stamp_size, n_bands)` depending on
  convention. The order of this array should correspond to the
  order in the returned `catalog`. Where `n_objects` is the
  number of detected objects by the algorithm. If you are using the multiresolution feature,
  you should instead return a dictionary with a key for each survey containing the
  aforementioned array.

* segmentation (np.ndarray): Array of booleans with shape `(n_objects,stamp_size,stamp_size)`
  The pixels set to True in the i-th channel correspond to the i-th
  object. The order should correspond to the order in the returned
  `catalog`. If you are using the multiresolution feature,
  you should instead return a dictionary with a key for each survey containing the
  aforementioned array.

Omitted keys in the returned dictionary are automatically assigned a `None` value (except for
`catalog` which is a mandatory entry).
"""
import os
from copy import deepcopy
from itertools import repeat
from typing import List

import astropy.table
import numpy as np
import sep
from astropy import units
from astropy.coordinates import SkyCoord
from skimage.feature import peak_local_max

from btk.multiprocess import multiprocess
from btk.utils import reverse_list_dictionary


def add_pixel_columns(catalog, wcs):
    """Uses the wcs to add a column to the catalog corresponding to pixel coordinates.

    The catalog must contain `ra` and `dec` columns.

    Args:
        catalog (astropy.table.Table): Catalog to modify.
        wcs (astropy.wcs.WCS): WCS corresponding to the wanted
                               transformation.
    """
    catalog_t = deepcopy(catalog)
    for blend in catalog_t:
        x_peak = []
        y_peak = []
        for gal in blend:
            coords = wcs.world_to_pixel_values(gal["ra"] / 3600, gal["dec"] / 3600)
            x_peak.append(coords[0])
            y_peak.append(coords[1])
        blend.add_column(x_peak, name="x_peak")
        blend.add_column(y_peak, name="y_peak")
    return catalog_t


def basic_measure(
    batch, idx, channels_last=False, surveys=None, is_multiresolution=False, **kwargs
):
    """Return centers detected with skimage.feature.peak_local_max.

    For each potentially multi-band image, an average over the bands is taken before measurement.
    NOTE: If this function is used with the multiresolution feature,
    measurements will be carried on the first survey.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.

    Returns:
            dict containing catalog with entries corresponding to measured peaks.
    """
    channel_indx = 0 if not channels_last else -1

    # multiresolution
    if is_multiresolution:
        if surveys is None:
            raise ValueError("surveys are required in order to use the MR feature.")
        survey_name = surveys[0].name
        avg_image = np.mean(batch["blend_images"][survey_name][idx], axis=channel_indx)
        wcs = batch["wcs"][survey_name]

    # single-survey
    else:
        avg_image = np.mean(batch["blend_images"][idx], axis=channel_indx)
        wcs = batch["wcs"]

    # set detection threshold to 5 times std of image
    threshold = 5 * np.std(avg_image)
    coordinates = peak_local_max(avg_image, min_distance=2, threshold_abs=threshold)

    # construct catalog from measurement.
    catalog = astropy.table.Table()
    catalog["ra"], catalog["dec"] = wcs.pixel_to_world_values(coordinates[:, 1], coordinates[:, 0])
    catalog["ra"] *= 3600
    catalog["dec"] *= 3600
    return {"catalog": catalog}


def sep_multiband_measure(
    batch,
    idx,
    channels_last=False,
    surveys=None,
    matching_threshold=1.0,
    sigma_noise=1.5,
    is_multiresolution=False,
    **kwargs,
):
    """Returns centers detected with source extractor by combining predictions in different bands.

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
    channel_indx = 0 if not channels_last else -1
    # multiresolution
    if is_multiresolution:
        if surveys is None:
            raise ValueError("surveys are required in order to use the MR feature.")
        survey_name = surveys[0].name
        image = batch["blend_images"][survey_name][idx]
        wcs = batch["wcs"][survey_name]

    # single-survey
    else:
        image = batch["blend_images"][idx]
        wcs = batch["wcs"]

    # run source extractor on the first band
    band_image = image[0] if channel_indx == 0 else image[:, :, 0]
    bkg = sep.Background(band_image)
    catalog = sep.extract(band_image, sigma_noise, err=bkg.globalrms, segmentation_map=False)

    # convert predictions to arcseconds
    ra_coordinates, dec_coordinates = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
    ra_coordinates *= 3600
    dec_coordinates *= 3600

    # iterate over remaining bands and match predictions using KdTree
    for band in range(1, image.shape[channel_indx]):
        # run source extractor
        band_image = image[band] if channel_indx == 0 else image[:, :, band]
        bkg = sep.Background(band_image)
        catalog = sep.extract(band_image, sigma_noise, err=bkg.globalrms, segmentation_map=False)

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
            idx, distance2d, _ = c1.match_to_catalog_sky(c2)
            distance2d = distance2d.arcsec

            # add new predictions, masking those that are closer than threshold
            ra_coordinates = np.concatenate(
                [ra_coordinates, ra_detections[distance2d > matching_threshold]]
            )
            dec_coordinates = np.concatenate(
                [dec_coordinates, dec_detections[distance2d > matching_threshold]]
            )
        else:
            ra_coordinates = np.concatenate([ra_coordinates, ra_detections])
            dec_coordinates = np.concatenate([dec_coordinates, dec_detections])

    # Wrap in the astropy table
    t = astropy.table.Table()
    t["ra"] = ra_coordinates
    t["dec"] = dec_coordinates

    return {"catalog": t}


def sep_singleband_measure(
    batch,
    idx,
    meas_band_num=3,
    use_mean=False,
    channels_last=False,
    surveys=None,
    sigma_noise=1.5,
    is_multiresolution=False,
    **kwargs,
):
    """Return detection, segmentation and deblending information running SEP on a single band.

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
    channel_indx = 0 if not channels_last else -1

    # multiresolution
    if is_multiresolution:
        if surveys is None:
            raise ValueError("surveys are required in order to use the MR feature.")
        survey_name = surveys[0].name
        image = batch["blend_images"][survey_name][idx]
        wcs = batch["wcs"][survey_name]
    # single-survey
    else:
        image = batch["blend_images"][idx]
        wcs = batch["wcs"]

    # get a 1-channel input for sep
    if use_mean:
        band_image = np.mean(image, axis=channel_indx)
    else:
        band_image = image[meas_band_num] if channel_indx == 0 else image[:, :, meas_band_num]

    # run source extractor
    stamp_size = band_image.shape[0]
    bkg = sep.Background(band_image)
    catalog, segmentation = sep.extract(
        band_image, sigma_noise, err=bkg.globalrms, segmentation_map=True
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
    t["ra"], t["dec"] = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
    t["ra"] *= 3600
    t["dec"] *= 3600

    # If multiresolution, return only the catalog
    if is_multiresolution:
        return {"catalog": t}
    else:
        return {
            "catalog": t,
            "segmentation": segmentation_exp,
            "deblended_images": deblended_images,
        }


class MeasureGenerator:
    """Generates output of deblender and measurement algorithm."""

    def __init__(
        self,
        measure_functions,
        draw_blend_generator,
        cpus=1,
        verbose=False,
        measure_kwargs: List[dict] = None,
        save_path=None,
    ):
        """Initialize measurement generator.

        Args:
            measure_functions: Function or list of functions that returns a dict with
                              measurements given output from the draw_blend_generator.
            draw_blend_generator: Generator that outputs dict with blended images,
                                  isolated images, blend catalog, wcs info, and psf.
            cpus: The number of parallel processes to run [Default: 1].
            verbose (bool): Whether to print information about measurement.
            measure_kwargs (list): List of dictionaries containing keyword arguments
                to be passed in to each of the `measure_functions`. Each dictionnary is
                passed one time to each function, meaning that each function which be
                ran as many times as there are different dictionnaries.
            save_path (str): Path to a directory where results will be saved. If left
                              as None, results will not be saved.
        """
        # setup and verify measure_functions.
        if callable(measure_functions):
            self.measure_functions = [measure_functions]
        elif isinstance(measure_functions, list):
            for f in measure_functions:
                if not callable(f):
                    raise ValueError(
                        "Your list 'measure_functions' does not consist of only functions."
                    )
            self.measure_functions = measure_functions
        else:
            ValueError("measure_functions must be a list of functions or a single function.")

        self.draw_blend_generator = draw_blend_generator
        self.cpus = cpus

        self.batch_size = self.draw_blend_generator.batch_size
        self.channels_last = self.draw_blend_generator.channels_last
        self.surveys = self.draw_blend_generator.surveys
        self.is_multiresolution = self.draw_blend_generator.is_multiresolution
        self.verbose = verbose
        self.save_path = save_path

        # initialize measure_kwargs dictionary.
        self.measure_kwargs = [{}] if measure_kwargs is None else measure_kwargs
        if not (
            isinstance(self.measure_kwargs, list)
            and np.all([isinstance(m, dict) for m in self.measure_kwargs])
        ):
            raise TypeError("measure_kwargs must be a list of dictionnaries.")
        for m in self.measure_kwargs:
            m["channels_last"] = self.channels_last
            m["surveys"] = self.surveys
            m["is_multiresolution"] = self.is_multiresolution

    def __iter__(self):
        """Return iterator which is the object itself."""
        return self

    def run_batch(self, batch, index, **kwargs):
        """Perform measurements on a single blend."""
        output = []
        for f in self.measure_functions:

            out = f(batch, index, **kwargs)

            # make sure output is in the correct format.
            if not isinstance(out["catalog"], astropy.table.Table):
                raise TypeError(
                    "The output dictionary of at least one of your measurement functions does not"
                    "contain an astropy table as the value of the key 'catalog'."
                )

            if not ("ra" in out["catalog"].colnames and "dec" in out["catalog"].colnames):
                raise ValueError(
                    "The output catalog of at least one of your measurement functions does not"
                    "contain the mandatory 'ra' and 'dec' columns"
                )

            for key in ["deblended_images", "segmentation"]:
                if key in out and out[key] is not None:
                    if len(self.surveys) == 1:
                        if not isinstance(out[key], np.ndarray):
                            raise TypeError(
                                f"The output '{key}' of at least one of your measurement"
                                f"functions is not a numpy array."
                            )
                        if key == "deblended_images":
                            if not out[key].shape[-3:] == batch["blend_images"].shape[-3:]:
                                raise ValueError(
                                    f"The shapes of the blended images in your {key} don't "
                                    f"match for at least one your measurement functions."
                                    f"{out[key].shape[-3:]} vs {batch['blend_images'].shape[-3:]}"
                                )
                    else:
                        for survey in self.surveys:
                            if not isinstance(out[key][survey.name], np.ndarray):
                                raise TypeError(
                                    f"The output '{key}' for survey '{survey.name}' of at least"
                                    f"one of your measurement functions is not a numpy array, but"
                                    f"a {type(out[key][survey.name])}"
                                )
                            if key == "deblended_images":
                                if (
                                    not out[key][survey.name].shape[-3:]
                                    == batch["blend_images"][survey.name].shape[-3:]
                                ):
                                    raise ValueError(
                                        f"The shapes of the blended images in your {key} for"
                                        f"survey '{survey.name}' do not match for at least one of"
                                        f"your measurement functions."
                                        f"{out[key].shape[-3:]} vs {batch['blend_images'].shape[-3:]}"  # noqa: E501
                                    )
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
        for f in self.measure_functions:
            for m in range(len(self.measure_kwargs)):
                key_name = f.__name__ + str(m) if len(self.measure_kwargs) > 1 else f.__name__
                catalog[key_name] = []
                segmentation[key_name] = []
                deblended_images[key_name] = []
        for m, measure_kwargs in enumerate(self.measure_kwargs):
            args_iter = ((blend_output, i) for i in range(self.batch_size))
            kwargs_iter = repeat(measure_kwargs)
            measure_output = multiprocess(
                self.run_batch,
                args_iter,
                kwargs_iter=kwargs_iter,
                cpus=self.cpus,
                verbose=self.verbose,
            )

            if self.verbose:
                print(f"Measurement {m} performed on batch")

            for i, f in enumerate(self.measure_functions):
                key_name = f.__name__ + str(m) if len(self.measure_kwargs) > 1 else f.__name__
                for j in range(len(measure_output)):
                    catalog[key_name].append(measure_output[j][i].get("catalog", None))
                    segmentation[key_name].append(measure_output[j][i].get("segmentation", None))
                    deblended_images[key_name].append(
                        measure_output[j][i].get("deblended_images", None)
                    )
                # If multiresolution, we reverse the order between the survey name and
                # the index of the blend
                if self.is_multiresolution:
                    survey_keys = list(blend_output["blend_list"].keys())
                    # We duplicate the catalog for each survey to get the pixel coordinates
                    catalogs_temp = {}
                    for surv in survey_keys:
                        catalogs_temp[surv] = add_pixel_columns(
                            catalog[key_name], blend_output["wcs"][surv]
                        )
                    catalog[key_name] = catalogs_temp

                    segmentation[key_name] = reverse_list_dictionary(
                        segmentation[key_name], survey_keys
                    )
                    deblended_images[key_name] = reverse_list_dictionary(
                        deblended_images[key_name], survey_keys
                    )

                else:
                    catalog[key_name] = add_pixel_columns(catalog[key_name], blend_output["wcs"])

                # save results if requested.
                if self.save_path is not None:

                    if not os.path.exists(os.path.join(self.save_path, key_name)):
                        os.mkdir(os.path.join(self.save_path, key_name))

                    if segmentation[key_name] is not None:
                        np.save(
                            os.path.join(self.save_path, key_name, "segmentation"),
                            np.array(segmentation[key_name], dtype=object),
                        )
                    if deblended_images[key_name] is not None:
                        np.save(
                            os.path.join(self.save_path, key_name, "deblended_images"),
                            np.array(deblended_images[key_name], dtype=object),
                        )
                    for j, cat in enumerate(catalog[key_name]):
                        cat.write(
                            os.path.join(self.save_path, key_name, f"detection_catalog_{j}"),
                            format="ascii",
                            overwrite=True,
                        )
        measure_results = {
            "catalog": catalog,
            "segmentation": segmentation,
            "deblended_images": deblended_images,
        }
        return blend_output, measure_results


available_measure_functions = {
    "basic": basic_measure,
    "sep_singleband_measure": sep_singleband_measure,
    "sep_multiband_measure": sep_multiband_measure,
}
