import multiprocessing as mp
from itertools import starmap


class Measurement_params(object):
    """Class describing functions to perform detection/deblending/measurement.
    """
    def make_measurement(self, data=None, index=None):
        return None

    def get_deblended_images(self, data=None, index=None):
        return None


def run_batch(Measurement_params, blend_output, index):
    deblend_results = Measurement_params.get_deblended_images(
        data=blend_output, index=index)
    measured_results = Measurement_params.make_measurement(
        data=blend_output, index=index)
    return [deblend_results, measured_results]


def generate(Measurement_params, draw_blend_generator, Args,
             multiprocessing=False, cpus=1):
    """Generates output of deblender and measurement algorithm.

    Args:
        Measurement_params: Class containing functions to perform deblending
                            and or measurement.
        draw_blend_generator: Generator that outputs dict with blended images,
                              isolated images, observing conditions and blend
                              catalog.
        Args: Class containing input parameters.
    Returns:
        draw_blend_generator output, deblender output and measurement output.
    """
    while True:
        blend_output = next(draw_blend_generator)
        batch_size = len(blend_output['blend_images'])
        deblend_results = {}
        measured_results = {}
        in_args = [(Measurement_params,
                    blend_output, i) for i in range(Args.batch_size)]
        print("in_args", len(in_args))
        if multiprocessing:
            if Args.verbose:
                print("Running mini-batch of size {0} with \
                    multiprocessing with pool {1}".format(len(in_args), cpus))
            with mp.Pool(processes=cpus) as pool:
                batch_results = pool.starmap(run_batch, in_args)
        else:
            if Args.verbose:
                print("Running mini-batch of size {0} \
                    serial {1} times".format(len(in_args), cpus))
            batch_results = list(starmap(run_batch, in_args))
        print("batch_results", batch_results)
        for i in range(batch_size):
            deblend_results.update(
                {i: batch_results[i][0]})
            measured_results.update(
                {i: batch_results[i][1]})
            if Args.verbose:
                print("Measurement performed on batch")
        yield blend_output, deblend_results, measured_results
