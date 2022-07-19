from unittest.mock import patch

from conftest import data_dir

from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.measure import MeasureGenerator, sep_multiband_measure, sep_singleband_measure
from btk.metrics import MetricsGenerator, meas_ksb_ellipticity
from btk.plot_utils import plot_metrics_summary
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys


@patch("btk.plot_utils.plt.show")
def test_multiresolution(mock_show):
    catalog_name = data_dir / "sample_input_catalog.fits"

    stamp_size = 24.0
    batch_size = 8
    cpus = 1
    add_noise = "all"
    surveys = get_surveys(["LSST", "HSC"])

    catalog = CatsimCatalog.from_file(catalog_name)
    sampling_function = DefaultSampling(stamp_size=stamp_size)
    draw_blend_generator = CatsimGenerator(
        catalog,
        sampling_function,
        surveys,
        stamp_size=stamp_size,
        batch_size=batch_size,
        cpus=cpus,
        add_noise=add_noise,
    )

    meas_generator = MeasureGenerator(
        [sep_singleband_measure, sep_multiband_measure], draw_blend_generator, cpus=cpus
    )
    metrics_generator = MetricsGenerator(
        meas_generator, target_meas={"ellipticity": meas_ksb_ellipticity}, meas_band_name=("r", "g")
    )
    blend_results, measure_results, metrics_results = next(metrics_generator)

    assert "LSST" in blend_results["blend_list"].keys(), "Both surveys get well defined outputs"
    assert "HSC" in blend_results["blend_list"].keys(), "Both surveys get well defined outputs"
    assert blend_results["blend_images"]["LSST"][0].shape[-1] == int(
        24.0 / 0.2
    ), "LSST survey should have a pixel scale of 0.2"
    assert blend_results["blend_images"]["HSC"][0].shape[-1] == int(
        24.0 / 0.168
    ), "HSC survey should have a pixel scale of 0.167"
    assert (
        "LSST" in measure_results["catalog"]["sep_singleband_measure"].keys()
    ), "Both surveys get well defined outputs"
    assert (
        "HSC" in measure_results["catalog"]["sep_singleband_measure"].keys()
    ), "Both surveys get well defined outputs"
    assert (
        "LSST" in metrics_results["galaxy_summary"]["sep_singleband_measure"].keys()
    ), "Both surveys get well defined outputs"
    assert (
        "HSC" in metrics_results["galaxy_summary"]["sep_singleband_measure"].keys()
    ), "Both surveys get well defined outputs"

    plot_metrics_summary(
        metrics_results,
        target_meas_keys=["ellipticity0"],
        target_meas_limits=[[-1, 1]],
        interactive=False,
    )
    plot_metrics_summary(
        metrics_results,
        target_meas_keys=["ellipticity0"],
        target_meas_limits=[[-1, 1]],
        interactive=True,
    )
