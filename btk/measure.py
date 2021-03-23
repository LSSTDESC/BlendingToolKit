"""Contains examples of functions that can be used to apply a measurement algorithm to the blends
 simulated by BTK. Every measurement function should take as an input a `batch` returned from a
 DrawBlendsGenerator object (see its `__next__` method) and an index corresponding to which image
 in the batch to measure.

It should return a dictionary containing a subset of the following keys/values (note the key
`catalog` is mandatory):
    - catalog (astropy.table.Table): An astropy table containing measurement information. The
                                     `len` of the table should be `n_objects`. If your
                                     DrawBlendsGenerator uses a single survey, the following
                                     column names are required:
                                        - x_peak: horizontal centroid position in pixels.
                                        - y_peak: vertical centroid position in pixels.
                                     For multiple surveys (multi-resolution), we instead require:
                                        - ra: object centroid right ascension in arcseconds,
                                        following the convention from the `wcs` object included in
                                        the input batch.
                                        - dec: vertical centroid position in arcseconds,
                                        following the convention from the `wcs` object included in
                                        the input batch.
    - deblended_image (np.ndarray): Array of deblended isolated images with shape:
                                    `(n_objects, n_bands, stamp_size, stamp_size)` or
                                    `(n_objects, stamp_size, stamp_size, n_bands)` depending on
                                    convention. The order of this array should correspond to the
                                    order in the returned `catalog`. Where `n_objects` is the
                                    number of detected objects
    - segmentation (np.ndarray): Array of booleans with shape `(n_objects,stamp_size,stamp_size)`
                                 The pixels set to True in the i-th channel correspond to the i-th
                                 object. The order should correspond to the order in the returned
                                 `catalog`.

Omitted keys in the returned dictionary are automatically assigned a `None` value (except for
`catalog` which is a mandatory entry).
"""
import astropy.table
import numpy as np
import sep
from skimage.feature import peak_local_max

from btk.multiprocess import multiprocess


def basic_measure(batch, idx):
    """Return centers detected when object detection is performed on the
    input image with skimage.feature.peak_local_max.

    NOTE: This function does not support the multi-resolution feature.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.

    Returns:
            dict containing catalog with entries corresponding to measured peaks.
    """
    if isinstance(batch["blend_images"], dict):
        raise NotImplementedError("This function does not support the multi-resolution feature.")

    coadd = np.mean(batch["blend_images"][idx], axis=0)

    # set detection threshold to 5 times std of image
    threshold = 5 * np.std(coadd)
    coordinates = peak_local_max(coadd, min_distance=2, threshold_abs=threshold)

    # construct catalog from measurement.
    catalog = astropy.table.Table()
    catalog["x_peak"] = coordinates[:, 1]
    catalog["y_peak"] = coordinates[:, 0]
    return {"catalog": catalog}


def sep_measure(batch, idx):
    """Return centers detected when object detection and photometry
    is done on input image with SEP.

    NOTE: This function does not support the multi-resolution feature.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.

    Returns:
        dict with the centers of sources detected by SEP detection algorithm.
    """
    if isinstance(batch["blend_images"], dict):
        raise NotImplementedError("This function does not support the multi-resolution feature.")

    image = batch["blend_images"][idx]
    stamp_size = image.shape[-2]  # true for both 'NCHW' or 'NHWC' formats.
    coadd = np.mean(image, axis=0)
    bkg = sep.Background(coadd)
    # Here the 1.5 value corresponds to a 1.5 sigma threshold for detection against noise.
    catalog, segmentation = sep.extract(coadd, 1.5, err=bkg.globalrms, segmentation_map=True)
    n_objects = len(catalog)
    segmentation_exp = np.zeros((n_objects, stamp_size, stamp_size), dtype=bool)
    deblended_images = np.zeros((n_objects, *image.shape), dtype=image.dtype)
    for i in range(n_objects):
        seg_i = segmentation == i + 1
        segmentation_exp[i] = seg_i
        deblended_images[i] = image * seg_i[np.newaxis, ...]

    t = astropy.table.Table()
    t["x_peak"] = catalog["x"]
    t["y_peak"] = catalog["y"]
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
        multiprocessing=False,
        cpus=1,
        verbose=False,
    ):
        """

        Args:
            measure_function: Function or list of functions that returns a dict with
                              measurements given output from the draw_blend_generator.
            draw_blend_generator: Generator that outputs dict with blended images,
                                  isolated images, blend catalog, wcs info, and psf.
            multiprocessing: If true performs multiprocessing of measurement.
            cpus: If multiprocessing is True, then number of parallel processes to
                 run [Default :1].
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
        self.multiprocessing = multiprocessing
        self.cpus = cpus

        self.batch_size = self.draw_blend_generator.batch_size
        self.dim_order = self.draw_blend_generator.dim_order

        self.verbose = verbose

    def __iter__(self):
        return self

    def run_batch(self, batch, index):
        output = []
        for f in self.measure_functions:

            out = f(batch, index)

            # make sure output is in the correct format.
            if not isinstance(out["catalog"], astropy.table.Table):
                raise TypeError(
                    "The output dictionary of at least one of your measurement functions does not"
                    "contain an astropy table as the value of the key 'catalog'."
                )

            if isinstance(batch["blend_images"], np.ndarray):
                if not (
                    "x_peak" in out["catalog"].colnames and "y_peak" in out["catalog"].colnames
                ):
                    raise ValueError(
                        "The output catalog of at least one of your measurement functions does not"
                        "contain the 'x_peak' and 'y_peak' columns which are mandatory for a single"
                        "survey study."
                    )

            if isinstance(batch["blend_images"], dict):
                if not ("ra" in out["catalog"].colnames and "dec" in out["catalog"].colnames):
                    raise ValueError(
                        "The output catalog of at least one of your measurement functions does not"
                        "contain the 'ra' and 'dec' columns which are mandatory for a"
                        "multi-resolution study."
                    )

            for key in ["deblended_images", "segmentation"]:
                if key in out and out[key] is not None:
                    if not isinstance(out[key], np.ndarray):
                        raise TypeError(
                            f"The output '{key}' of at least one of your measurement"
                            f"functions is not a numpy array."
                        )
                    if not out[key].shape[-2:] == batch["blend_images"].shape[-2:]:
                        raise ValueError(
                            f"The shapes of the blended images in your {key} don't"
                            f"match for at least one your measurement functions."
                        )

            out = {k: out.get(k, None) for k in self.measure_params}
            output.append(out)
        return output

    def __next__(self):
        """
        Returns:
            draw_blend_generator output from `__next__` method.
            measurement output: List of length `batch_size`, where each element is a list of
                                `len(measure_functions)` corresponding to the measurements made by
                                each function on each element of the batch.
        """

        blend_output = next(self.draw_blend_generator)
        input_args = ((blend_output, i) for i in range(self.batch_size))
        measure_results = multiprocess(
            self.run_batch,
            input_args,
            self.cpus,
            self.multiprocessing,
            self.verbose,
        )
        if self.verbose:
            print("Measurement performed on batch")
        return blend_output, measure_results
