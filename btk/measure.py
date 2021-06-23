"""File containing measurement infrastructure for the BlendingToolKit.

Contains examples of functions that can be used to apply a measurement algorithm to the blends
 simulated by BTK. Every measurement function should have the following skeleton:

 ```
def measure_function(batch, idx, **kwargs):
    # do some measurements on the images contained in batch.
    return output
 ```

where `batch` is the output from the `DrawBlendsGenerator` object (see its `__next__` method) and
`idx` is the index corresponding to which image in the batch to measure. The additional keyword
arguments `**kwargs` can be passed via the `measure_kwargs` dictionary argument in the
`MeasureGerator` initialize which are shared among all the measurement functions.

It should return a dictionary containing a subset of the following keys/values (note the key
`catalog` is mandatory):

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
  `catalog`.

Omitted keys in the returned dictionary are automatically assigned a `None` value (except for
`catalog` which is a mandatory entry).
"""
import os
from itertools import repeat

import astropy.table
import numpy as np
import sep
from skimage.feature import peak_local_max

from btk.multiprocess import multiprocess


def basic_measure(batch, idx, channels_last=False, **kwargs):
    """Return centers detected with skimage.feature.peak_local_max.

    NOTE: This function does not support the multi-resolution feature.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.

    Returns:
            dict containing catalog with entries corresponding to measured peaks.
    """
    channel_indx = 0 if not channels_last else -1
    if isinstance(batch["blend_images"], dict):
        surveys = kwargs.get("surveys", None)
        survey_name = surveys[0].name
        coadd = np.mean(batch["blend_images"][survey_name][idx], axis=channel_indx)
        wcs = batch["wcs"][survey_name]
    else:
        coadd = np.mean(batch["blend_images"][idx], axis=channel_indx)
        wcs = batch["wcs"]

    # set detection threshold to 5 times std of image
    threshold = 5 * np.std(coadd)
    coordinates = peak_local_max(coadd, min_distance=2, threshold_abs=threshold)

    # construct catalog from measurement.
    catalog = astropy.table.Table()
    # catalog["x_peak"], catalog["y_peak"] = coordinates[:, 1], coordinates[:, 0]
    catalog["ra"], catalog["dec"] = wcs.pixel_to_world_values(coordinates[:, 1], coordinates[:, 0])
    return {"catalog": catalog}


def sep_measure(batch, idx, channels_last=False, **kwargs):
    """Return detection, segmentation and deblending information with SEP.

    NOTE: This function does not support the multi-resolution feature.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.

    Returns:
        dict with the centers of sources detected by SEP detection algorithm.
    """
    sigma_noise = kwargs.get("sigma_noise", 1.5)
    # Here the 1.5 value corresponds to a 1.5 sigma threshold for detection against noise.

    channel_indx = 0 if not channels_last else -1
    if isinstance(batch["blend_images"], dict):
        surveys = kwargs.get("surveys", None)
        survey_name = surveys[0].name
        image = batch["blend_images"][survey_name][idx]
        coadd = np.mean(image, axis=channel_indx)
        wcs = batch["wcs"][survey_name]
    else:
        image = batch["blend_images"][idx]
        coadd = np.mean(image, axis=channel_indx)
        wcs = batch["wcs"]

    stamp_size = coadd.shape[0]
    bkg = sep.Background(coadd)
    catalog, segmentation = sep.extract(
        coadd, sigma_noise, err=bkg.globalrms, segmentation_map=True
    )
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

    t = astropy.table.Table()
    # t["x_peak"], t["y_peak"] = catalog["x"], catalog["y"]
    t["ra"], t["dec"] = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
    if isinstance(batch["blend_images"], dict):  # If multiresolution, return only the catalog
        return {"catalog": t}
    else:
        return {
            "catalog": t,
            "segmentation": segmentation_exp,
            "deblended_images": deblended_images,
        }


class MeasureGenerator:
    """Generates output of deblender and measurement algorithm.

    Attributes:
        self.measure_functions (list): List of functions that take as input the output from
                                        DrawBlendsGenerator and return the output of a measurement
                                        (see module docstring).
    """

    measure_params = {"deblended_images", "catalog", "segmentation"}

    def __init__(
        self,
        measure_functions,
        draw_blend_generator,
        cpus=1,
        verbose=False,
        measure_kwargs: list = None,
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
            measure_kwargs (list): list of dictionaries containing keyword arguments
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
        self.verbose = verbose
        self.save_path = save_path

        # initialize measure_kwargs dictionary.
        self.measure_kwargs = [{}] if measure_kwargs is None else measure_kwargs
        for m in self.measure_kwargs:
            m["channels_last"] = self.channels_last
            m["surveys"] = self.draw_blend_generator.surveys

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
                    "contain the 'ra' and 'dec' columns which are mandatory for a"
                    "multi-resolution study."
                )

            # for key in ["deblended_images", "segmentation"]:
            #     if key in out and out[key] is not None:
            #         if not isinstance(out[key], np.ndarray):
            #             raise TypeError(
            #                 f"The output '{key}' of at least one of your measurement"
            #                 f"functions is not a numpy array."
            #             )
            #         if key == "deblended_images":
            #             if not out[key].shape[-3:] == batch["blend_images"].shape[-3:]:
            #                 raise ValueError(
            #                     f"The shapes of the blended images in your {key} don't "
            #                     f"match for at least one your measurement functions."
            #                     f"{out[key].shape[-3:]} vs {batch['blend_images'].shape[-3:]}"
            #                 )

            out = {k: out.get(k, None) for k in self.measure_params}
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
                if isinstance(blend_output["blend_list"], dict):
                    survey_keys = list(blend_output["blend_list"].keys())
                    if segmentation[key_name][0] is None:
                        segmentation[key_name] = {
                            k: [None for n in range(len(segmentation[key_name]))]
                            for k in survey_keys
                        }
                    else:
                        segmentation[key_name] = {
                            k: [
                                segmentation[key_name][n][k]
                                for n in range(len(segmentation[key_name]))
                            ]
                            for k in survey_keys
                        }
                    if deblended_images[key_name][0] is None:
                        deblended_images[key_name] = {
                            k: [None for n in range(len(deblended_images[key_name]))]
                            for k in survey_keys
                        }
                    else:
                        deblended_images[key_name] = {
                            k: [
                                deblended_images[key_name][n][k]
                                for n in range(len(deblended_images[key_name]))
                            ]
                            for k in survey_keys
                        }

                # save results if requested.
                if self.save_path is not None:

                    if not os.path.exists(os.path.join(self.save_path, key_name)):
                        os.mkdir(os.path.join(self.save_path, key_name))

                    if segmentation[key_name] is not None:
                        np.save(
                            os.path.join(self.save_path, key_name, "segmentation"),
                            segmentation[key_name],
                        )
                    if deblended_images[key_name] is not None:
                        np.save(
                            os.path.join(self.save_path, key_name, "deblended_images"),
                            deblended_images[key_name],
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


available_measure_functions = {"basic": basic_measure, "sep": sep_measure}
