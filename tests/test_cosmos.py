import btk
from btk.survey import Survey

SEED = 0


def test_cosmos_generator(data_dir):
    """Test the pipeline as a whole for a single deblender."""
    cosmos_catalog_paths = [
        data_dir / "cosmos" / "real_galaxy_catalog_23.5_example.fits",
        data_dir / "cosmos" / "real_galaxy_catalog_23.5_example_fits.fits",
    ]
    cosmos_catalog_files = [p.as_posix() for p in cosmos_catalog_paths]
    catalog = btk.catalog.CosmosCatalog.from_file(cosmos_catalog_files)

    _ = catalog.get_raw_catalog()

    survey: Survey = btk.survey.get_surveys("LSST")
    fltr = survey.get_filter("r")
    assert hasattr(fltr, "psf")

    stamp_size = 24.0
    max_shift = 1.0
    max_n_sources = 2
    sampling_function = btk.sampling_functions.DefaultSampling(
        max_number=max_n_sources,
        min_number=1,
        stamp_size=stamp_size,
        max_shift=max_shift,
        min_mag=20,
        max_mag=21,
        seed=SEED,
        mag_name="MAG",
    )

    batch_size = 10

    draw_generator = btk.draw_blends.CosmosGenerator(
        catalog,
        sampling_function,
        survey,
        batch_size=batch_size,
        njobs=1,
        add_noise="all",
        seed=SEED,
        gal_type="real",
    )

    # generate batch 100 blend catalogs and images.
    blend_batch = next(draw_generator)
    assert len(blend_batch.catalog_list) == batch_size
    assert blend_batch.blend_images.shape == (batch_size, 6, stamp_size / 0.2, stamp_size / 0.2)
    iso_shape = (batch_size, max_n_sources, 6, stamp_size / 0.2, stamp_size / 0.2)
    assert blend_batch.isolated_images.shape == iso_shape
