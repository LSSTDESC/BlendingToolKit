from conftest import data_dir

from btk.catalog import CosmosCatalog
from btk.draw_blends import CosmosGenerator
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys

COSMOS_CATALOG_PATHS = [
    str(data_dir / "cosmos/real_galaxy_catalog_23.5_example.fits"),
    str(data_dir / "cosmos/real_galaxy_catalog_23.5_example_fits.fits"),
]

COSMOS_EXT_CATALOG_PATHS = [
    str(data_dir / "cosmos/real_galaxy_catalog_26_extension_example.fits"),
    str(data_dir / "cosmos/real_galaxy_catalog_26_extension_example_fits.fits"),
]


def test_cosmos_galaxies():
    stamp_size = 24.0
    batch_size = 2
    catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, exclusion_level="none")
    sampling_function = DefaultSampling(stamp_size=stamp_size)
    HST = get_surveys("HST")

    draw_generator = CosmosGenerator(
        catalog,
        sampling_function,
        HST,
        batch_size=batch_size,
        stamp_size=stamp_size,
        cpus=1,
        add_noise="all",
        verbose=True,
    )

    _ = next(draw_generator)


def test_cosmos_ext_galaxies():
    stamp_size = 24.0
    batch_size = 2
    catalog = CosmosCatalog.from_file(COSMOS_EXT_CATALOG_PATHS, exclusion_level="none")
    sampling_function = DefaultSampling(stamp_size=stamp_size)
    HST = get_surveys("HST")

    draw_generator = CosmosGenerator(
        catalog,
        sampling_function,
        HST,
        batch_size=batch_size,
        stamp_size=stamp_size,
        cpus=1,
        add_noise="all",
        verbose=True,
    )

    _ = next(draw_generator)
