import pytest

import btk.sampling_functions
from btk.survey import Rubin


def test_sampling_no_max_number():
    catalog_name = "data/sample_input_catalog.fits"

    class TestSamplingFunction(btk.sampling_functions.SamplingFunction):
        def __init__(self):
            pass

        def __call__(self, table, **kwargs):
            pass

        @property
        def compatible_catalogs(self):
            return "CatsimCatalog", "CosmosCatalog"

    with pytest.raises(AttributeError) as excinfo:
        catalog_name = "data/sample_input_catalog.fits"

        stamp_size = 24.0
        batch_size = 8
        cpus = 1
        multiprocessing = False
        add_noise = True

        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
        sampling_function = TestSamplingFunction()
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog,
            sampling_function,
            [Rubin],
            stamp_size=stamp_size,
            batch_size=batch_size,
            multiprocessing=multiprocessing,
            cpus=cpus,
            add_noise=add_noise,
            meas_bands=("i"),
        )
        draw_output = next(draw_generator)  # noqa: F841

    assert "max_number" in str(excinfo.value)


def test_sampling_incompatible_catalog():
    # FAILING
    catalog_name = "data/sample_input_catalog.fits"

    class TestSamplingFunction(btk.sampling_functions.SamplingFunction):
        def __call__(self, table, **kwargs):
            pass

        @property
        def compatible_catalogs(self):
            return "CosmosCatalog"

    with pytest.raises(AttributeError) as excinfo:
        catalog_name = "data/sample_input_catalog.fits"

        stamp_size = 24.0
        batch_size = 8
        cpus = 1
        multiprocessing = False
        add_noise = True

        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
        sampling_function = TestSamplingFunction(max_number=5)
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog,
            sampling_function,
            [Rubin],
            stamp_size=stamp_size,
            batch_size=batch_size,
            multiprocessing=multiprocessing,
            cpus=cpus,
            add_noise=add_noise,
            meas_bands=("i"),
        )
        draw_output = next(draw_generator)  # noqa: F841

    assert "Your catalog and sampling functions are not compatible with each other." in str(
        excinfo.value
    )


def test_sampling_too_much_objects():
    # FAILING
    catalog_name = "data/sample_input_catalog.fits"

    class TestSamplingFunction(btk.sampling_functions.SamplingFunction):
        def __call__(self, table, **kwargs):
            return table[: self.max_number + 1]

        @property
        def compatible_catalogs(self):
            return "CatsimCatalog", "CosmosCatalog"

    with pytest.raises(ValueError) as excinfo:
        catalog_name = "data/sample_input_catalog.fits"

        stamp_size = 24.0
        batch_size = 8
        cpus = 1
        multiprocessing = False
        add_noise = True

        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
        sampling_function = TestSamplingFunction(max_number=5)
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog,
            sampling_function,
            [Rubin],
            stamp_size=stamp_size,
            batch_size=batch_size,
            multiprocessing=multiprocessing,
            cpus=cpus,
            add_noise=add_noise,
            meas_bands=("i"),
        )
        draw_output = next(draw_generator)  # noqa: F841

    assert "Number of objects per blend must be less than max_number" in str(excinfo.value)


# def test_multiresolution():
#     catalog_name = "data/sample_input_catalog.fits"

#     stamp_size = 24.0
#     batch_size = 8
#     cpus = 1
#     multiprocessing = False
#     add_noise = True

#     catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
#     sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)
#     draw_generator = btk.draw_blends.CatsimGenerator(
#         catalog,
#         sampling_function,
#         [Rubin, HSC],
#         stamp_size=stamp_size,
#         batch_size=batch_size,
#         multiprocessing=multiprocessing,
#         cpus=cpus,
#         add_noise=add_noise,
#         meas_bands=("i", "i"),
#     )
#     draw_output = next(draw_generator)
