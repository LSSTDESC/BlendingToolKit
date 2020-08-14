class Measurement_params(object):
    """Class describing functions to perform detection/deblending/measurement.
    """

    def make_measurement(self, data=None, index=None):
        return None

    def get_deblended_images(self, data=None, index=None):
        return None


def generate(Measurement_params, draw_blend_generator, Args):
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
        batch_size = len(blend_output["blend_images"])
        deblend_results = {}
        measured_results = {}
        for i in range(batch_size):
            deblend_results.update(
                {i: Measurement_params.get_deblended_images(data=blend_output, index=i)}
            )
            measured_results.update(
                {i: Measurement_params.make_measurement(data=blend_output, index=i)}
            )
            if Args.verbose:
                print("Measurement performed on batch")
        yield blend_output, deblend_results, measured_results
