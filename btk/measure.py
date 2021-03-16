"""Contains examples of functions that can be used to apply a measurement algorithm to the blends
 simulated by BTK. Every measurement function should take as an input a batch returned from a
 DrawBlendsGenerator object (see its `__next__` method) and an index corresponding to which image
 in the batch to measure.

It should return a dictionary containing a subset of the following keys/values:
    - catalog (astropy.table.Table): An astropy table containing measurement information. The
                                     `len` of the table should be `n_objects`. Currently the
                                     following column names are supported:
                                        - dx: horizontal centroid position in pixels.
                                        - dy: vertical centroid positoin in pixels.
                                     This key/value pair is mandatory, as well as the columns
                                     `dx` and `dy`.
    - deblended_image (np.ndarray): Array of deblended isolated images with shape:
                                    `(n_objects, n_bands, stamp_size, stamp_size)`. The order
                                    should correspond to the order in the returned `catalog`.
    - segmentation (np.ndarray): Array of booleans with shape `(n_objects,stamp_size,stamp_size)`
                                 The pixels set to True in the i-th channel correspond to the i-th
                                 object. The order should correspond to the order in the returned
                                 `catalog`.
where `n_objects` is the number of detected objects. Omitted keys in the returned dictionary are
automatically assigned a `None` value (except for `catalog` which is a mandatory entry).
"""
import astropy
import numpy as np
import sep
from skimage.feature import peak_local_max

from btk.multiprocess import multiprocess


def basic_measure(batch, idx):
    """Return centers detected when object detection is performed on the
    input image with skimage.feature.peak_local_max.

    NOTE: Assumes at least 3 bands are present in the images produced.

    Args:
        batch (dict): Output of DrawBlendsGenerator object's `__next__` method.
        idx (int): Index number of blend scene in the batch to preform
            measurement on.

    Returns:
            dict containing catalog with entries corresponding to measured peaks.
    """
    if isinstance(batch["blend_images"], dict):
        raise NotImplementedError("This function does not support the multi-resolution feature.")

    image = np.mean(batch["blend_images"][idx], axis=0)

    # set detection threshold to 5 times std of image
    threshold = 5 * np.std(image)
    coordinates = peak_local_max(image, min_distance=2, threshold_abs=threshold)

    # construct catalog from measurement.
    catalog = astropy.table.Table()
    catalog["dx"] = coordinates[:, 1]
    catalog["dy"] = coordinates[:, 0]
    return {"catalog": catalog}


def sep_measure(batch, idx):
    """Return centers detected when object detection and photometry
    is done on input image with SEP.

    NOTE: Assumes at least 3 bands are present in the images produced.

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
    catalog, segmentation = sep.extract(coadd, 1.5, err=bkg.globalrms, segmentation_map=True)
    n_objects = len(catalog)
    segmentation_exp = np.zeros((n_objects, stamp_size, stamp_size))
    deblended_images = np.zeros((n_objects, *image.shape))
    for i in range(n_objects):
        segmentation_exp[i][np.where(segmentation == i + 1)] = True
        deblended_images[i] = segmentation[i].reshape(1, stamp_size, stamp_size) * image

    # TODO: maybe change to using 'ra', 'dec' looking towards MR feature. We can use WCS for this.
    #  Or make it optional if not using MR.
    # construct astropy table
    t = astropy.table.Table()
    t["dx"] = catalog["x"]
    t["dy"] = catalog["y"]
    return {
        "catalog": t,
        "segmentation": segmentation_exp,
        "deblended_images": deblended_images,
    }


class MeasureGenerator:
    measure_params = {"deblended_images", "peaks", "segmentation"}

    def __init__(
        self,
        measure_functions,
        draw_blend_generator,
        multiprocessing=False,
        cpus=1,
        verbose=False,
    ):
        """Generates output of deblender and measurement algorithm.

        Args:
            measure_function: Function or list of functions that returns a dict with
                              measurements given output from the draw_blend_generator.
            draw_blend_generator: Generator that outputs dict with blended images,
                                  isolated images, blend catalog, wcs info, and psf.
            multiprocessing: If true performs multiprocessing of measurement.
            cpus: If multiprocessing is True, then number of parallel processes to
                 run [Default :1].
        """
        # setup measure_functions
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

        self.verbose = verbose

    def __iter__(self):
        return self

    def run_batch(self, batch, index):
        # get some parameters for error-checking.
        # stamp_size = batch["blend_images"].shape[-2]  # true for both 'NCHW' or 'NHWC' formats.
        # max_sources = max(len(t) for t in batch["blend_list"])

        output = []
        for f in self.measure_functions:

            out = f(batch, index)

            # make sure output is in the correct format.
            if not isinstance(out["catalog"], astropy.table.table.Table):
                raise TypeError(
                    "The output dictionary of at least one of your measurement functions does not"
                    "contain an astropy table as the value of the key 'catalog'."
                )

            if not ("dx" in out["catalog"].colnames and "dy" in out["catalog"].colnames):
                raise ValueError(
                    "The output catalog of at least one of your measurement functions does not"
                    "contain the 'dx' and 'dy' columns which are mandatory."
                )

            for key in ["deblended_images", "segmentation"]:
                if key in out and out[key] is not None:
                    if not isinstance(out[key], np.ndarray):
                        TypeError(
                            f"The output '{key}' of at least one of your measurement"
                            f"functions is not a numpy array."
                        )
                    if (
                        not out[key].shape[-1] == batch["blend_images"][index].shape[-1]
                        or not out[key].shape[-2] == batch["blend_images"][index].shape[-2]
                    ):
                        pass

            out = {k: out.get(k, None) for k in self.measure_params}
            output.append(out)
        return output

    def __next__(self):
        """
        Returns:
            draw_blend_generator output
            measurement output: List of len(batch_size), where each element is a list of
                                len(measure_functions) corresponding to the measurements made by
                                each function on each element of the batch. If only one
                                measure_function was passed in, then each sub-list is converted
                                to a single element.
        """

        blend_output = next(self.draw_blend_generator)
        input_args = [(blend_output, i) for i in range(self.batch_size)]
        measure_results = multiprocess(
            self.run_batch,
            input_args,
            self.cpus,
            self.multiprocessing,
            self.verbose,
        )
        for i, r in enumerate(measure_results):
            if len(r) == 1:
                measure_results[i] = r[0]
        if self.verbose:
            print("Measurement performed on batch")
        return blend_output, measure_results
