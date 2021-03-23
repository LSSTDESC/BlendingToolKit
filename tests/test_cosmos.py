import btk.catalog
import btk.draw_blends
import btk.survey

COSMOS_CATALOG_PATHS = [
    "data/real_galaxy_catalog_25.2.fits",
    "data/real_galaxy_catalog_25.2_fits.fits",
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
        multiprocessing=False,
        cpus=1,
        add_noise=True,
        verbose=True,
        meas_bands=["f814w"],
    )

    _ = next(draw_generator)
