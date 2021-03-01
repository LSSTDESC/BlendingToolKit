import btk
from btk.survey import HSC
from btk.survey import Rubin


def test_multiresolution():
    catalog_name = "data/sample_input_catalog.fits"

    stamp_size = 24.0
    batch_size = 8
    cpus = 1
    multiprocessing = False
    add_noise = True

    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)
    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        [Rubin, HSC],
        stamp_size=stamp_size,
        batch_size=batch_size,
        multiprocessing=multiprocessing,
        cpus=cpus,
        add_noise=add_noise,
        meas_bands=("i", "i"),
    )
    draw_output = next(draw_generator)

    assert "LSST" in draw_output["blend_list"].keys(), "Both surveys get well defined outputs"
    assert "HSC" in draw_output["blend_list"].keys(), "Both surveys get well defined outputs"
    assert draw_output["blend_images"]["LSST"][0].shape[0] == int(
        24.0 / 0.2
    ), "LSST survey should have a pixel scale of 0.2"
    assert draw_output["blend_images"]["HSC"][0].shape[0] == int(
        24.0 / 0.167
    ), "HSC survey should have a pixel scale of 0.167"
