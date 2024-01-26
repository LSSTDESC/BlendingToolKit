"""Test pipeline as a whole at a high-level."""

import tempfile

import numpy as np

import btk
from btk.survey import Survey

SEED = 0


def test_pipeline(data_dir):
    """Test the pipeline as a whole for a single deblender."""
    catalog_file = data_dir / "input_catalog.fits"
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_file)

    _ = catalog.get_raw_catalog()

    survey: Survey = btk.survey.get_surveys("LSST")
    fltr = survey.get_filter("r")
    assert hasattr(fltr, "psf")

    # single bright galaxy
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
    )

    batch_size = 10

    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        survey,
        batch_size=batch_size,
        stamp_size=stamp_size,
        njobs=1,
        add_noise="all",
        seed=SEED,
    )

    # generate batch 100 blend catalogs and images.
    blend_batch = next(draw_generator)
    assert len(blend_batch.catalog_list) == batch_size
    assert blend_batch.blend_images.shape == (batch_size, 6, stamp_size / 0.2, stamp_size / 0.2)
    iso_shape = (batch_size, max_n_sources, 6, stamp_size / 0.2, stamp_size / 0.2)
    assert blend_batch.isolated_images.shape == iso_shape

    # test deblender with SEP runs
    deblender = btk.deblend.SepSingleBand(
        max_n_sources=max_n_sources,  # same as above
        thresh=3,  # threshold pixel value for detection (see SEP docs)
        use_band=2,  # measure on 'r' band
    )
    _ = deblender.deblend(0, blend_batch)  # deblend the first blend in the batch.
    deblend_batch = deblender(blend_batch, njobs=1)  # deblend the whole batch.
    assert len(deblend_batch.catalog_list) == batch_size
    assert deblend_batch.segmentation.shape == (
        batch_size,
        max_n_sources,
        stamp_size / 0.2,
        stamp_size / 0.2,
    )
    deblend_shape = (batch_size, max_n_sources, 1, stamp_size / 0.2, stamp_size / 0.2)
    assert deblend_batch.deblended_images.shape == deblend_shape

    # test matching works
    matcher = btk.match.PixelHungarianMatcher(pixel_max_sep=5.0)

    true_catalog_list = blend_batch.catalog_list
    pred_catalog_list = deblend_batch.catalog_list
    matching = matcher(true_catalog_list, pred_catalog_list)  # matching object
    matched_true_catalogs = matching.match_true_catalogs(true_catalog_list)
    matched_pred_catalogs = matching.match_pred_catalogs(pred_catalog_list)

    for tcat, pcat in zip(matched_true_catalogs, matched_pred_catalogs):
        assert len(tcat) == len(pcat)

    # detection
    recall = btk.metrics.detection.Recall(batch_size)
    precision = btk.metrics.detection.Precision(batch_size)
    tp, t, p = matching.tp, matching.n_true, matching.n_pred
    assert recall(tp, t, p) > 0.60
    assert precision(tp, t, p) > 0.60

    # reconstruction
    mse = btk.metrics.reconstruction.MSE(batch_size)
    iso_images1 = blend_batch.isolated_images[:, :, 2]  # only r-band
    iso_images2 = deblend_batch.deblended_images[:, :, 0]
    iso_images_matched1 = matching.match_true_arrays(iso_images1)
    iso_images_matched2 = matching.match_pred_arrays(iso_images2)
    mse(iso_images_matched1, iso_images_matched2)

    # test saving
    with tempfile.TemporaryDirectory() as tmpdirname:
        blend_batch.save(tmpdirname, 0)
        blend_batch2 = btk.blend_batch.BlendBatch.load(tmpdirname, 0)

        deblend_batch.save(tmpdirname, 0)
        deblend_batch2 = btk.blend_batch.DeblendBatch.load(tmpdirname, 0)

        assert blend_batch.batch_size == blend_batch2.batch_size
        assert blend_batch.stamp_size == blend_batch2.stamp_size

        assert deblend_batch.batch_size == deblend_batch2.batch_size
        assert deblend_batch.image_size == deblend_batch2.image_size


def test_sep(data_dir):
    """Check we always detect single bright objects."""

    catalog_file = data_dir / "input_catalog.fits"
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_file)
    survey: Survey = btk.survey.get_surveys("LSST")

    # single bright galaxy=
    sampling_function = btk.sampling_functions.DefaultSampling(
        max_number=1,
        min_number=1,
        stamp_size=24.0,
        max_shift=1.0,
        min_mag=0,
        max_mag=21,
        seed=SEED,
    )

    assert np.sum((catalog.table["i_ab"] > 0) & (catalog.table["i_ab"] < 21)) > 100

    batch_size = 100

    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        survey,
        batch_size=batch_size,
        stamp_size=24.0,
        njobs=1,
        add_noise="all",
        seed=SEED,
    )

    blend_batch = next(draw_generator)
    deblender = btk.deblend.SepSingleBand(max_n_sources=1, thresh=3, use_band=2)
    deblend_batch = deblender(blend_batch, njobs=1)

    matcher = btk.match.PixelHungarianMatcher(pixel_max_sep=5.0)

    true_catalog_list = blend_batch.catalog_list
    pred_catalog_list = deblend_batch.catalog_list
    matching = matcher(true_catalog_list, pred_catalog_list)  # matching object
    tp, t, p = matching.tp, matching.t, matching.p

    recall = btk.metrics.detection.Recall(batch_size)
    precision = btk.metrics.detection.Precision(batch_size)

    assert recall(tp, t, p) > 0.95
    assert precision(tp, t, p) > 0.95


def test_scarlet(data_dir):
    """Check scarlet deblender implementation runs without too many failures."""

    max_n_sources = 3
    stamp_size = 24.0
    seed = 0
    max_shift = 2.0  # shift is only 2 arcsecs -> 10 pixels, so blends are likely.

    catalog = btk.catalog.CatsimCatalog.from_file(data_dir / "input_catalog.fits")
    sampling_function = btk.sampling_functions.DefaultSampling(
        max_number=max_n_sources,
        min_number=max_n_sources,  # always 3 sources in every blend.
        stamp_size=stamp_size,
        max_shift=max_shift,
        min_mag=24,
        max_mag=25,
        seed=seed,
    )
    LSST = btk.survey.get_surveys("LSST")

    batch_size = 10

    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        LSST,
        batch_size=batch_size,
        stamp_size=stamp_size,
        njobs=1,
        add_noise="all",
        seed=seed,  # use same seed here
    )

    blend_batch = next(draw_generator)
    deblender = btk.deblend.Scarlet(max_n_sources)
    deblend_batch = deblender(blend_batch, reference_catalogs=blend_batch.catalog_list)
    n_failures = np.sum([len(cat) == 0 for cat in deblend_batch.catalog_list], axis=0)
    assert n_failures <= 3


def test_density_sampling(data_dir):
    """Test new density sampling function."""
    catalog = btk.catalog.CatsimCatalog.from_file(data_dir / "input_catalog.fits")

    sampling_function = btk.sampling_functions.DensitySampling(
        max_number=50,
        min_number=1,
        stamp_size=24,
        max_shift=10,
        min_mag=-np.inf,
        max_mag=27.3,
        seed=SEED,
    )

    for _ in range(10):
        blends = sampling_function(catalog.get_raw_catalog())

        assert 1 <= len(blends) <= 50
        assert np.all(-10 <= blends["ra"]) & np.all(blends["ra"] <= 10)
        assert np.all(-10 <= blends["dec"]) & np.all(blends["dec"] <= 10)
