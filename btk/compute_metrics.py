"""For an input measure generator and measurement_function algorithm
this script will:
1) compare the output from measure to truth
2) return perfomance metrics

Performance metrics computed separately for detection, segmentation, flux,
redshift

"""


class Metrics_params(object):
    def evaluate_detection(self, data=None, index=None):
        return None

    def evaluate_segmentation(self, data=None, index=None):
        return None

    def evaluate_flux(self, data=None, index=None):
        return None

    def evalute_redshift(self, data=None, index=None):
        return None


def generate(Metrics_params, measure_generator, Args):
    while True:
        meas_output = next(measure_generator)
        # meas_output = blend_output, deblend_results, measured_results
        batch_size = len(meas_output[0]['blend_images'])
        results = {}
        for i in range(batch_size):
            results[i] = {}
            results[i]['dectection'] = Metrics_params.evaluate_detection(
                data=meas_output, index=i)
            results[i]['segmentation'] = Metrics_params.evaluate_segmentation(
                data=meas_output, index=i)
            results[i]['flux'] = Metrics_params.evaluate_flux(
                data=meas_output, index=i)
            results[i]['redshift'] = Metrics_params.evaluate_redshift(
                data=meas_output, index=i)
        yield results, meas_output
