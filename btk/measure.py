"""For an input draw_object generator and measurement_function algorithm
this script will:
1) draw blended and isolated objects
2) Run the deblending ans measuremnt algorithm as defined by the
Measuremnt_args class

"""


class Measurement_params(object):
    def make_measurement(self, data=None, index=None):
        return None

    def get_deblended_images(self, data=None, index=None):
        return None


def generate(Measurement_params, draw_blend_generator, Args):
    while True:
        blend_output = next(draw_blend_generator)
        batch_size = len(blend_output['blend_images'])
        deblend_results = []
        measured_results = []
        for i in range(batch_size):
            deblend_results.append(
                Measurement_params.get_deblended_images(data=blend_output,
                                                        index=i))
            measured_results.append(
                Measurement_params.make_measurement(data=blend_output,
                                                    index=i))
            if Args.verbose:
                print("Measurement performed on batch")
        yield blend_output, deblend_results, measured_results
