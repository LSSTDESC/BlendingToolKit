from conftest import data_dir

import btk.catalog
import btk.draw_blends
import btk.survey

COSMOS_CATALOG_PATHS = [
    data_dir / "cosmos/real_galaxy_catalog_23.5_example.fits",
    data_dir / "cosmos/real_galaxy_catalog_23.5_example_fits.fits",
]


def test_cosmos_galaxies():
    stamp_size = 24.0
    batch_size = 2
    catalog = btk.catalog.CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
    sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size)

    draw_generator = btk.draw_blends.CosmosGenerator(
        catalog,
        sampling_function,
        [btk.survey.HST],
        batch_size=batch_size,
        stamp_size=stamp_size,
        cpus=1,
        add_noise=True,
        verbose=True,
        meas_bands=["f814w"],
    )

    _ = next(draw_generator)
