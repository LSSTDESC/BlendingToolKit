import pytest
from conftest import data_dir

from btk.catalog import CatsimCatalog, CosmosCatalog
from btk.draw_blends import (
    CatsimGenerator,
    CosmosGenerator,
    SourceNotVisible,
    get_catsim_galaxy,
)
from btk.sampling_functions import DefaultSampling, SamplingFunction
from btk.survey import (
    Filter,
    get_default_psf,
    get_default_psf_with_galcheat_info,
    get_psf_from_file,
    get_surveys,
)

CATALOG_PATH = data_dir / "sample_input_catalog.fits"

COSMOS_CATALOG_PATHS = [
    str(data_dir / "cosmos/real_galaxy_catalog_23.5_example.fits"),
    str(data_dir / "cosmos/real_galaxy_catalog_23.5_example_fits.fits"),
]

COSMOS_EXT_CATALOG_PATHS = [
    str(data_dir / "cosmos/real_galaxy_catalog_26_extension_example.fits"),
    str(data_dir / "cosmos/real_galaxy_catalog_26_extension_example_fits.fits"),
]


def test_sampling_no_max_number():
    class TestSamplingFunction(SamplingFunction):
        def __init__(self):
            pass

        def __call__(self, table, **kwargs):
            pass

        @property
        def compatible_catalogs(self):
            return "CatsimCatalog", "CosmosCatalog"

    with pytest.raises(AttributeError) as excinfo:
        stamp_size = 24.0
        batch_size = 8
        cpus = 1
        add_noise = "all"

        catalog = CatsimCatalog.from_file(CATALOG_PATH)
        sampling_function = TestSamplingFunction()
        draw_generator = CatsimGenerator(
            catalog,
            sampling_function,
            get_surveys("LSST"),
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
        )
        draw_output = next(draw_generator)  # noqa: F841

    assert "max_number" in str(excinfo.value)


def test_sampling_incompatible_catalog():
    class TestSamplingFunction(SamplingFunction):
        def __call__(self, table, **kwargs):
            pass

        @property
        def compatible_catalogs(self):
            return "CosmosCatalog"

    with pytest.raises(AttributeError) as excinfo:
        stamp_size = 24.0
        batch_size = 8
        cpus = 1
        add_noise = "all"

        catalog = CatsimCatalog.from_file(CATALOG_PATH)
        sampling_function = TestSamplingFunction(max_number=5)
        draw_generator = CatsimGenerator(
            catalog,
            sampling_function,
            get_surveys("LSST"),
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
        )
        draw_output = next(draw_generator)  # noqa: F841

    assert "Your catalog and sampling functions are not compatible with each other." in str(
        excinfo.value
    )


def test_sampling_too_much_objects():
    # FAILING
    CATALOG_PATH = "data/sample_input_catalog.fits"

    class TestSamplingFunction(SamplingFunction):
        def __call__(self, table, **kwargs):
            return table[: self.max_number + 1]

        @property
        def compatible_catalogs(self):
            return "CatsimCatalog", "CosmosCatalog"

    with pytest.raises(ValueError) as excinfo:
        stamp_size = 24.0
        batch_size = 8
        cpus = 1
        add_noise = "all"

        catalog = CatsimCatalog.from_file(CATALOG_PATH)
        sampling_function = TestSamplingFunction(max_number=5)
        draw_generator = CatsimGenerator(
            catalog,
            sampling_function,
            get_surveys("LSST"),
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
        )
        next(draw_generator)  # noqa: F841

    assert "Number of objects per blend must be less than max_number" in str(excinfo.value)


def test_source_not_visible():
    survey = get_surveys("LSST")
    filt = Filter.from_dict(
        dict(
            name="u",
            psf_fwhm=0.859,
            zeropoint=9.16,
            sky_brightness=22.9,
            full_exposure_time=1680,
            effective_wavelength=3592.13,
        )
    )
    filt.psf = get_default_psf_with_galcheat_info(survey, filt)
    catalog = CatsimCatalog.from_file(CATALOG_PATH)
    with pytest.raises(SourceNotVisible):
        get_catsim_galaxy(catalog.table[0], filt, survey, True, True, True)


def test_survey_not_list():
    stamp_size = 24.0
    batch_size = 8
    cpus = 1
    add_noise = True

    catalog = CatsimCatalog.from_file(CATALOG_PATH)
    sampling_function = DefaultSampling(stamp_size=stamp_size)
    with pytest.raises(TypeError):
        draw_generator = CatsimGenerator(
            catalog,
            sampling_function,
            3,
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
        )
        next(draw_generator)


def test_psf():
    get_default_psf(
        mirror_diameter=8.36,
        effective_area=32.4,
        filt_wavelength=7528.51,
        fwhm=0.748,
        atmospheric_model="Moffat",
    )
    get_default_psf(
        mirror_diameter=8.36,
        effective_area=32.4,
        filt_wavelength=7528.51,
        fwhm=0.748,
        atmospheric_model=None,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        get_default_psf(
            mirror_diameter=8.36,
            effective_area=32.4,
            filt_wavelength=7528.51,
            fwhm=0.748,
            atmospheric_model="Layered",
        )

    assert "atmospheric model request" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        get_default_psf(mirror_diameter=1, effective_area=4, filt_wavelength=7528.51, fwhm=0.748)

    assert "Incompatible effective-area and mirror-diameter values." in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        get_default_psf(
            mirror_diameter=0,
            effective_area=0,
            filt_wavelength=7528.51,
            fwhm=0.748,
            atmospheric_model=None,
        )

    assert "Neither the atmospheric nor the optical PSF components are defined." in str(
        excinfo.value
    )

    get_default_psf(mirror_diameter=0, effective_area=0, filt_wavelength=7528.51, fwhm=0.748)

    get_psf_from_file("tests/example_psf", get_surveys("LSST"))
    get_psf_from_file("tests/multi_psf", get_surveys("LSST"))
    # The case where the folder is empty cannot be tested as you cannot add an empty folder to git


def test_incompatible_catalogs():
    stamp_size = 24.0
    batch_size = 8
    cpus = 1
    add_noise = True

    catalog = CatsimCatalog.from_file(CATALOG_PATH)
    sampling_function = DefaultSampling(stamp_size=stamp_size)
    with pytest.raises(ValueError):
        # Wrong generator
        draw_generator = CosmosGenerator(  # noqa: F841
            catalog,
            sampling_function,
            get_surveys("LSST"),
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
        )
    with pytest.raises(ValueError):
        # Missing filter
        CatsimGenerator(
            catalog,
            sampling_function,
            get_surveys("COSMOS"),
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
        )

    catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, exclusion_level="none")
    with pytest.raises(ValueError):
        CatsimGenerator(
            catalog,
            sampling_function,
            get_surveys("LSST"),
            stamp_size=stamp_size,
            batch_size=batch_size,
            cpus=cpus,
            add_noise=add_noise,
        )
