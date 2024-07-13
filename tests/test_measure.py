"""Test measure functions run on simple outputs from generator and deblenders."""

import numpy as np
from surveycodex.utilities import mean_sky_level

import btk
from btk.measure import get_aperture_fluxes, get_blendedness, get_ksb_ellipticity, get_snr
from btk.survey import Survey

SEED = 0


def test_measure(data_dir):
    catalog_file = data_dir / "input_catalog.fits"
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_file)

    _ = catalog.get_raw_catalog()

    survey: Survey = btk.survey.get_surveys("LSST")
    fltr = survey.get_filter("r")
    assert hasattr(fltr, "psf")

    stamp_size = 24.0
    max_shift = 2.0
    max_n_sources = 4
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

    batch = next(draw_generator)
    sky_level = mean_sky_level(survey, survey.get_filter("r")).to_value("electron")

    # combine all centroids
    xs_peak = np.zeros((batch_size, max_n_sources))
    ys_peak = np.zeros((batch_size, max_n_sources))
    for ii, t in enumerate(batch.catalog_list):
        n_sources = len(t["x_peak"])
        xs_peak[ii, :n_sources] = t["x_peak"].value
        ys_peak[ii, :n_sources] = t["y_peak"].value
    centroids = np.concatenate(
        [xs_peak.reshape(batch_size, -1, 1), ys_peak.reshape(batch_size, -1, 1)], axis=2
    )

    # aperture photometry
    fluxes, fluxerr = get_aperture_fluxes(batch.blend_images[:, 2], xs_peak, ys_peak, 5, sky_level)
    assert fluxes.shape == (batch_size, max_n_sources)
    assert fluxerr.shape == (batch_size, max_n_sources)

    # blendedness
    blendedness = get_blendedness(batch.isolated_images[:, :, 2])
    assert blendedness.shape == (batch_size, max_n_sources)
    assert np.all(np.less_equal(blendedness, 1)) and np.all(np.greater_equal(blendedness, 0.0))

    # snr
    snr = get_snr(batch.isolated_images[:, :, 2], sky_level)
    snr.shape == (batch_size, max_n_sources)
    assert np.all(np.greater_equal(snr, 0))

    # ellipticity
    ellips = get_ksb_ellipticity(batch.isolated_images[:, :, 2], centroids, batch.psf[2], 0.2)
    assert ellips.shape == (batch_size, max_n_sources, 2)
    assert np.sum(np.abs(ellips) == 10) == 0  # should output np.nan for bad shear measurements

    # zeroes if no galaxies
    for ii in range(batch_size):
        n_sources = len(batch.catalog_list[ii])
        for jj in range(max_n_sources):
            if jj >= n_sources:
                assert snr[ii, jj] == 0
                assert np.all(np.isnan(ellips[ii, jj]))
                assert blendedness[ii, jj] == 0
