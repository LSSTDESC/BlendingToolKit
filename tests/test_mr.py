from conftest import data_dir

from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys


def test_multiresolution():
    catalog_name = data_dir / "sample_input_catalog.fits"

    stamp_size = 24.0
    batch_size = 8
    cpus = 1
    add_noise = True
    surveys = get_surveys(["Rubin", "HSC"])

    catalog = CatsimCatalog.from_file(catalog_name)
    sampling_function = DefaultSampling(stamp_size=stamp_size)
    draw_generator = CatsimGenerator(
        catalog,
        sampling_function,
        surveys,
        stamp_size=stamp_size,
        batch_size=batch_size,
        cpus=cpus,
        add_noise=add_noise,
    )
    draw_output = next(draw_generator)

    assert "Rubin" in draw_output["blend_list"].keys(), "Both surveys get well defined outputs"
    assert "HSC" in draw_output["blend_list"].keys(), "Both surveys get well defined outputs"
    assert draw_output["blend_images"]["Rubin"][0].shape[-1] == int(
        24.0 / 0.2
    ), "Rubin survey should have a pixel scale of 0.2"
    assert draw_output["blend_images"]["HSC"][0].shape[-1] == int(
        24.0 / 0.167
    ), "HSC survey should have a pixel scale of 0.167"
