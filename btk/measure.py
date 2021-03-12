"""Contains examples of functions that can be used to apply a measurement algorithm to the blends
 simulated by BTK. Every measurement function should take as an input a batch returned from a
 DrawBlendsGenerator object (see its `__next__` method) and an index corresponding to which image
 in the batch to measure.

It should return a dictionary containing a subset of the following keys/values:
    - deblended_image (np.ndarray): Array of deblended isolated images with shape:
                                  [batch_size, max_sources, n_bands, nx, ny]
    - peaks (np.ndarray): Array of predicted centroids in pixels.
                          Shape: [batch_size, max_sources, 2].

Omitted entries are automatically assigned a `None` value.
"""
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
            dict containing subset of keys from: ['deblend_image', 'peaks']
    """

    image = np.mean(batch["blend_images"][idx], axis=0)

    # set detection threshold to 5 times std of image
    threshold = 5 * np.std(image)
    coordinates = peak_local_max(image, min_distance=2, threshold_abs=threshold)
    peaks = np.stack((coordinates[:, 1], coordinates[:, 0]), axis=1)
    return {"deblend_image": None, "peaks": peaks}


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

    image = np.mean(batch["blend_images"][idx], axis=0)
    bkg = sep.Background(image)
    catalog = sep.extract(image, 1.5, err=bkg.globalrms, segmentation_map=False)
    centers = np.stack((catalog["x"], catalog["y"]), axis=1)
    return {"deblended_image": None, "peaks": centers}


class MeasureGenerator:
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
            ValueError("measure_functions must be a list of functions or a function.")

        self.draw_blend_generator = draw_blend_generator
        self.multiprocessing = multiprocessing
        self.cpus = cpus

        self.batch_size = self.draw_blend_generator.batch_size

        self.verbose = verbose

    def __iter__(self):
        return self

    def run_batch(self, batch, index):
        output = []
        for f in self.measure_functions:
            output.append(f(batch, index))
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
