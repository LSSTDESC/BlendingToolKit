import numpy as np

import btk
from btk.survey import Survey

SEED = 0


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
        njobs=1,
        add_noise="all",
        seed=seed,  # use same seed here
    )

    blend_batch = next(draw_generator)
    deblender = btk.deblend.Scarlet(max_n_sources)
    deblend_batch = deblender(blend_batch, reference_catalogs=blend_batch.catalog_list)
    n_failures = np.sum([len(cat) == 0 for cat in deblend_batch.catalog_list], axis=0)
    assert n_failures <= 3
