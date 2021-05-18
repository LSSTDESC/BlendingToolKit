import tempfile

import numpy as np

import btk
from btk.survey import Rubin


def test_save():
    output_dir = tempfile.mkdtemp()
    catalog_name = "data/sample_input_catalog.fits"
    stamp_size = 24.0
    batch_size = 8
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)
    draw_blend_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        [Rubin],
        batch_size=batch_size,
        stamp_size=stamp_size,
        save_path=output_dir,
    )
    meas_generator = btk.measure.MeasureGenerator(
        btk.measure.sep_measure, draw_blend_generator, save_path=output_dir
    )
    metrics_generator = btk.metrics.MetricsGenerator(
        meas_generator,
        use_metrics=("detection", "segmentation", "reconstruction"),
        target_meas={"ellipticity": btk.metrics.meas_ksb_ellipticity},
        save_path=output_dir,
    )
    blend_results, measure_results, metrics_results = next(metrics_generator)
    blend_results2, measure_results2, metrics_results2 = btk.utils.load_all_results(
        output_dir, ["LSST"], ["sep_measure"], batch_size
    )
    np.testing.assert_array_equal(
        blend_results["blend_images"], blend_results2["blend_images"]["LSST"]
    )
    np.testing.assert_array_equal(
        measure_results["sep_measure"]["segmentation"][0],
        measure_results2["sep_measure"]["segmentation"][0],
    )
    np.testing.assert_array_equal(
        measure_results["sep_measure"]["deblended_images"][0],
        measure_results2["sep_measure"]["deblended_images"][0],
    )
    np.testing.assert_array_equal(
        metrics_results["sep_measure"]["galaxy_summary"]["distance_closest_galaxy"],
        metrics_results2["sep_measure"]["galaxy_summary"]["distance_closest_galaxy"],
    )
