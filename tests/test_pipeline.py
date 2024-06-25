"""Test pipeline as a whole at a high-level."""

import tempfile

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
