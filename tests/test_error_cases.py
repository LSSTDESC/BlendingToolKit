import pytest

import btk
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
        add_noise = True

        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
        sampling_function = TestSamplingFunction()
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog,
            sampling_function,
            [Rubin],
            stamp_size=stamp_size,
            batch_size=batch_size,
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
        add_noise = True

        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
        sampling_function = TestSamplingFunction(max_number=5)
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog,
            sampling_function,
            [Rubin],
            stamp_size=stamp_size,
            batch_size=batch_size,
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
        add_noise = True

        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
        sampling_function = TestSamplingFunction(max_number=5)
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog,
            sampling_function,
            [Rubin],
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
            meas_bands=("i"),
        )
        draw_output = next(draw_generator)  # noqa: F841

    assert "Number of objects per blend must be less than max_number" in str(excinfo.value)


def test_source_not_visible():
    filt = btk.survey.Filter(
        name="u",
        psf=btk.survey.get_psf(
            mirror_diameter=8.36,
            effective_area=32.4,
            filt_wavelength=3592.13,
            fwhm=0.859,
        ),
        sky_brightness=22.9,
        exp_time=1680,
        zeropoint=9.16,
        extinction=0.451,
    )
    catalog_name = "data/sample_input_catalog.fits"
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    with pytest.raises(btk.draw_blends.SourceNotVisible):
        gal = btk.draw_blends.get_catsim_galaxy(  # noqa: F841
            catalog.table[0], filt, Rubin, True, True, True
        )


def test_survey_not_list():
    catalog_name = "data/sample_input_catalog.fits"

    stamp_size = 24.0
    batch_size = 8
    cpus = 1
    add_noise = True

    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)
    with pytest.raises(TypeError):
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog,
            sampling_function,
            3,
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
            meas_bands=("i"),
        )
        draw_output = next(draw_generator)  # noqa: F841


def test_psf():
    btk.survey.get_psf(
        mirror_diameter=8.36,
        effective_area=32.4,
        filt_wavelength=7528.51,
        fwhm=0.748,
        atmospheric_model="Moffat",
    )
    btk.survey.get_psf(
        mirror_diameter=8.36,
        effective_area=32.4,
        filt_wavelength=7528.51,
        fwhm=0.748,
        atmospheric_model=None,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        btk.survey.get_psf(
            mirror_diameter=8.36,
            effective_area=32.4,
            filt_wavelength=7528.51,
            fwhm=0.748,
            atmospheric_model="Layered",
        )

    assert "atmospheric model request" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        btk.survey.get_psf(mirror_diameter=1, effective_area=4, filt_wavelength=7528.51, fwhm=0.748)

    assert "Incompatible effective-area and mirror-diameter values." in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        btk.survey.get_psf(
            mirror_diameter=0,
            effective_area=0,
            filt_wavelength=7528.51,
            fwhm=0.748,
            atmospheric_model=None,
        )

    assert "Neither the atmospheric nor the optical PSF components are defined." in str(
        excinfo.value
    )

    btk.survey.get_psf(mirror_diameter=0, effective_area=0, filt_wavelength=7528.51, fwhm=0.748)

    btk.survey.get_psf_from_file("tests/example_psf", Rubin)
    btk.survey.get_psf_from_file("tests/multi_psf", Rubin)
    # The case where the folder is empty cannot be tested as you cannot add an empty folder to git
