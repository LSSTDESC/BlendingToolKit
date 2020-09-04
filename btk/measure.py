from abc import ABC
from btk.multiprocess import multiprocess


class Measurement_params(ABC):
    """Class with functions to perform detection/deblending/measurement."""

    def make_measurement(self, data, index):
        """Function describing how the measurement algorithm is run.

        Args:
            data (dict): Output generated by btk.draw_blends containing blended
                         images, isolated images, observing conditions and
                         blend catalog, for a given batch.
            index (int): Index number of blend scene in the batch to preform
                         measurement on.

        Returns:
            output of measurement algorithm (fluxes, shapes, size, etc.) as
            an astropy catalog.
        """
        return None

    def get_deblended_images(self, data, index):
        """Function describing how the deblending algorithm is run.

        Args:
            data (dict): Output generated by btk.draw_blends containing blended
                         images, isolated images, observing conditions and
                         blend catalog, for a given batch.
            index (int): Index number of blend scene in the batch to preform
                         measurement on.

        Returns:
            output of deblending algorithm as a dict.
        """
        return None


class MeasureGenerator:
    def __init__(
        self,
        measurement_params,
        draw_blend_generator,
        multiprocessing=False,
        cpus=1,
        verbose=False,
    ):
        """Generates output of deblender and measurement algorithm.

        Args:
            measurement_params: Instance from class
                                `btk.measure.Measurement_params`.
            draw_blend_generator: Generator that outputs dict with blended images,
                                  isolated images, observing conditions and blend
                                  catalog.
            multiprocessing: If true performs multiprocessing of measurement.
            cpus: If multiprocessing is True, then number of parallel processes to
                 run [Default :1].
        Returns:
            draw_blend_generator output, deblender output and measurement output.
        """
        self.measurement_params = measurement_params
        self.draw_blend_generator = draw_blend_generator
        self.multiprocessing = multiprocessing
        self.cpus = cpus

        self.batch_size = self.draw_blend_generator.batch_size

        self.verbose = verbose

    def __iter__(self):
        return self

    def run_batch(self, blend_output, index):
        deblend_results = self.measurement_params.get_deblended_images(
            data=blend_output, index=index
        )
        measured_results = self.measurement_params.make_measurement(
            data=blend_output, index=index
        )
        return [deblend_results, measured_results]

    def __next__(self):

        blend_output = next(self.draw_blend_generator)
        deblend_results = {}
        measured_results = {}
        input_args = [(blend_output, i) for i in range(self.batch_size)]
        batch_results = multiprocess(
            self.run_batch,
            input_args,
            self.cpus,
            self.multiprocessing,
            self.verbose,
        )
        for i in range(self.batch_size):
            deblend_results.update({i: batch_results[i][0]})
            measured_results.update({i: batch_results[i][1]})
        if self.verbose:
            print("Measurement performed on batch")
        return blend_output, deblend_results, measured_results
