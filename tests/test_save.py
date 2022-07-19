import tempfile

import numpy as np

from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.measure import MeasureGenerator, sep_singleband_measure
from btk.metrics import MetricsGenerator, meas_ksb_ellipticity
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys
from btk.utils import load_all_results


def test_save():
    output_dir = tempfile.mkdtemp()
    catalog_name = "data/sample_input_catalog.fits"
    stamp_size = 24.0
    batch_size = 8
    catalog = CatsimCatalog.from_file(catalog_name)
    sampling_function = DefaultSampling(stamp_size=stamp_size)
    draw_blend_generator = CatsimGenerator(
        catalog,
        sampling_function,
        get_surveys("LSST"),
        batch_size=batch_size,
        stamp_size=stamp_size,
        save_path=output_dir,
    )
    meas_generator = MeasureGenerator(
        sep_singleband_measure, draw_blend_generator, save_path=output_dir
    )
    metrics_generator = MetricsGenerator(
        meas_generator,
        target_meas={"ellipticity": meas_ksb_ellipticity},
        save_path=output_dir,
    )
    blend_results, measure_results, metrics_results = next(metrics_generator)
    blend_results2, measure_results2, metrics_results2 = load_all_results(
        output_dir, ["LSST"], ["sep_singleband_measure"], batch_size
    )
    np.testing.assert_array_equal(
        blend_results["blend_images"], blend_results2["blend_images"]["LSST"]
    )
    np.testing.assert_array_equal(
        measure_results["segmentation"]["sep_singleband_measure"][0],
        measure_results2["segmentation"]["sep_singleband_measure"][0],
    )
    np.testing.assert_array_equal(
        measure_results["deblended_images"]["sep_singleband_measure"][0],
        measure_results2["deblended_images"]["sep_singleband_measure"][0],
    )
    np.testing.assert_array_equal(
        metrics_results["galaxy_summary"]["sep_singleband_measure"]["distance_closest_galaxy"],
        metrics_results2["galaxy_summary"]["sep_singleband_measure"]["distance_closest_galaxy"],
    )
