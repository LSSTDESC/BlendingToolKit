"""We have this unittests to avoid running the very time consuming advanced notebook."""

import multiprocessing as mp

import numpy as np

import btk


def get_psf_size(survey: btk.survey.Survey) -> float:
    """Return the PSF size in pixels."""
    psf_size_arcsec = survey.get_filter("r").psf_fwhm.to_value("arcsec")
    pixel_scale = survey.pixel_scale.to_value("arcsec")
    return psf_size_arcsec / pixel_scale


def _setup_generator(data_dir):
    max_n_sources = 10
    min_n_sources = 0
    stamp_size = 24.0
    max_shift = 3.0  # shift from center is 3 arcsecs = 15 pixels, so blends are likely.
    seed = 0

    catalog = btk.catalog.CatsimCatalog.from_file(data_dir / "input_catalog.fits")

    sampling_function = btk.sampling_functions.DefaultSampling(
        max_number=max_n_sources,
        min_number=min_n_sources,
        stamp_size=stamp_size,
        max_shift=max_shift,
        min_mag=18,
        max_mag=27,
        mag_name="i_ab",  # cutting on i-band
        seed=seed,
    )

    survey = btk.survey.get_surveys("LSST")

    batch_size = 10

    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        survey,
        batch_size=batch_size,
        njobs=1,
        add_noise="background",
        seed=seed,  # use same seed here
    )

    return {
        "draw_generator": draw_generator,
        "survey": survey,
        "max_n_sources": max_n_sources,
        "batch_size": batch_size,
    }


def test_efficiency_matrix(data_dir):
    from surveycodex.utilities import mean_sky_level

    from btk.deblend import PeakLocalMax, SepSingleBand
    from btk.match import PixelHungarianMatcher
    from btk.metrics.detection import Efficiency

    setup_dict = _setup_generator(data_dir)
    draw_generator = setup_dict["draw_generator"]
    survey = setup_dict["survey"]
    max_n_sources = setup_dict["max_n_sources"]
    batch_size = setup_dict["batch_size"]

    # sky level
    sky_level = mean_sky_level(survey, survey.get_filter("r")).to_value("electron")  # gain = 1

    # use psf size as minimum distance between peaks (in pixels) for the peak-finding algorithm.
    min_distance = int(get_psf_size(survey))  # needs to be an integer

    # standard values for SEP that work well for blended galaxy scenes
    thresh = 1.5
    min_area = 3

    # setup both deblenders
    peak_finder = PeakLocalMax(
        max_n_sources=max_n_sources + 10,
        sky_level=sky_level,
        threshold_scale=5,
        min_distance=min_distance * 2,
        use_band=2,  # r-band
    )

    sep = SepSingleBand(
        max_n_sources=max_n_sources + 10, thresh=thresh, min_area=min_area, use_band=2
    )

    # matcher
    matcher = PixelHungarianMatcher(pixel_max_sep=min_distance)

    # setup efficiency matrix metric
    eff_matrix_peak = Efficiency(batch_size)
    eff_matrix_sep = Efficiency(batch_size)

    for _ in range(2):
        blend_batch = next(draw_generator)
        peak_batch = peak_finder(blend_batch)
        sep_batch = sep(blend_batch)
        matching_peak = matcher(blend_batch.catalog_list, peak_batch.catalog_list)
        matching_sep = matcher(blend_batch.catalog_list, sep_batch.catalog_list)
        eff_matrix_peak(matching_peak.tp, matching_peak.t, matching_peak.p)
        eff_matrix_sep(matching_sep.tp, matching_sep.t, matching_sep.p)

    # get efficiency matrices and normalize
    _ = eff_matrix_peak.aggregate()
    _ = eff_matrix_sep.aggregate()


def test_recall_curves(data_dir):
    from surveycodex.utilities import mean_sky_level

    from btk.deblend import PeakLocalMax, SepSingleBand

    setup_dict = _setup_generator(data_dir)
    draw_generator = setup_dict["draw_generator"]
    survey = setup_dict["survey"]
    max_n_sources = setup_dict["max_n_sources"]
    batch_size = setup_dict["batch_size"]

    # sky level
    sky_level = mean_sky_level(survey, survey.get_filter("r")).to_value("electron")  # gain = 1

    # use psf size as minimum distance between peaks (in pixels).
    min_distance = int(get_psf_size(survey))  # needs to be an integer

    # setup both deblenders
    peak_finder = PeakLocalMax(
        max_n_sources=max_n_sources + 10,
        sky_level=sky_level,
        threshold_scale=5,
        min_distance=min_distance * 2,
        use_band=2,  # r-band
    )

    sep = SepSingleBand(max_n_sources=max_n_sources + 10, thresh=1.5, min_area=3, use_band=2)

    from btk.match import PixelHungarianMatcher

    # matcher
    matcher = PixelHungarianMatcher(pixel_max_sep=min_distance)

    snr_bins = np.linspace(0, 100, 21)

    from btk.measure import get_snr
    from btk.metrics.detection import Recall

    # we create one recall metric object per bin
    # each of them will automatically aggregate results over batches
    recalls_peaks = [Recall(batch_size) for _ in range(1, len(snr_bins))]
    recalls_sep = [Recall(batch_size) for _ in range(1, len(snr_bins))]

    for _ in range(2):
        blend_batch = next(draw_generator)
        iso_images = blend_batch.isolated_images[:, :, 2]  # pick 'r' band
        snr_r = get_snr(iso_images, sky_level)

        # run deblenders and matches
        peak_batch = peak_finder(blend_batch)
        sep_batch = sep(blend_batch)
        matching_peak = matcher(blend_batch.catalog_list, peak_batch.catalog_list)
        matching_sep = matcher(blend_batch.catalog_list, sep_batch.catalog_list)

        for jj in range(1, len(snr_bins)):
            min_snr, _ = snr_bins[jj - 1], snr_bins[jj]
            mask = snr_r > min_snr
            matching_peak_new = matching_peak.filter_by_true(mask)
            matching_sep_new = matching_sep.filter_by_true(mask)
            recalls_peaks[jj - 1](matching_peak_new.tp, matching_peak_new.t, matching_peak_new.p)
            recalls_sep[jj - 1](matching_sep_new.tp, matching_sep_new.t, matching_sep_new.p)

    _ = np.array([recall.aggregate() for recall in recalls_peaks])
    _ = np.array([recall.aggregate() for recall in recalls_sep])


def test_reconstruction_histograms(data_dir):
    from btk.deblend import Scarlet, SepSingleBand
    from btk.match import PixelHungarianMatcher
    from btk.metrics.reconstruction import MSE, PSNR, StructSim

    setup_dict = _setup_generator(data_dir)
    draw_generator = setup_dict["draw_generator"]
    survey = setup_dict["survey"]
    max_n_sources = setup_dict["max_n_sources"]
    batch_size = setup_dict["batch_size"]

    metrics_sep = {"mse": MSE(batch_size), "psnr": PSNR(batch_size), "ssim": StructSim(batch_size)}

    metrics_scarlet = {
        "mse": MSE(batch_size),
        "psnr": PSNR(batch_size),
        "ssim": StructSim(batch_size),
    }

    # same as before
    thresh = 1.5
    min_area = 3

    # use psf size as minimum distance between peaks (in pixels).
    min_distance = int(get_psf_size(survey))
    sep = SepSingleBand(max_n_sources=max_n_sources, thresh=thresh, use_band=2, min_area=min_area)
    scarlet = Scarlet(max_n_sources)
    matcher = PixelHungarianMatcher(min_distance)

    njobs = 4 if mp.cpu_count() > 4 else mp.cpu_count() - 1

    for ii in range(2):
        blend_batch = next(draw_generator)
        sep_batch = sep(blend_batch)
        scarlet_batch = scarlet(
            blend_batch,  # this line takes a while
            reference_catalogs=sep_batch.catalog_list,
            njobs=njobs,
        )
        matching_sep = matcher(blend_batch.catalog_list, sep_batch.catalog_list)
        matching_scarlet = matcher(blend_batch.catalog_list, scarlet_batch.catalog_list)

        true_iso_images = blend_batch.isolated_images[:, :, 2]  # pick 'r' band
        iso_images_sep = sep_batch.deblended_images[
            :, :, 0
        ]  # pick the only band which is the 'r' band
        iso_images_scarlet = scarlet_batch.deblended_images[:, :, 2]  # pick 'r' band

        iso_images1 = matching_sep.match_true_arrays(true_iso_images)
        iso_images2 = matching_scarlet.match_true_arrays(true_iso_images)
        iso_images_sep = matching_sep.match_pred_arrays(iso_images_sep)
        iso_images_scarlet = matching_scarlet.match_pred_arrays(iso_images_scarlet)

        for metric in metrics_sep.values():
            metric(iso_images1, iso_images_sep)

        for metric in metrics_scarlet.values():
            metric(iso_images2, iso_images_scarlet)

    # join data from all batches into single array

    # sep
    all_sep = {"mse": np.array([]), "psnr": np.array([]), "ssim": np.array([])}
    for metric_name, metric in metrics_sep.items():
        for mvalues in metric.all_data:
            all_sep[metric_name] = np.concatenate([all_sep[metric_name], mvalues[metric_name]])

    # scarlet
    all_scarlet = {"mse": np.array([]), "psnr": np.array([]), "ssim": np.array([])}
    for metric_name, metric in metrics_scarlet.items():
        for mvalues in metric.all_data:
            all_scarlet[metric_name] = np.concatenate(
                [all_scarlet[metric_name], mvalues[metric_name]]
            )


def test_ellipticity_residuals(data_dir):
    from surveycodex.utilities import mean_sky_level

    from btk.deblend import Scarlet
    from btk.match import PixelHungarianMatcher
    from btk.measure import get_blendedness, get_ksb_ellipticity, get_snr

    setup_dict = _setup_generator(data_dir)
    draw_generator = setup_dict["draw_generator"]
    survey = setup_dict["survey"]
    max_n_sources = setup_dict["max_n_sources"]

    # we will continue using 'r' band
    sky_level = mean_sky_level(survey, survey.get_filter("r")).to_value("electron")  # gain = 1

    # use psf size as minimum distance between peaks (in pixels).
    min_distance = int(get_psf_size(survey))
    scarlet = Scarlet(max_n_sources)
    matcher = PixelHungarianMatcher(min_distance)

    es1 = []
    es2 = []
    snrs = []
    bs = []

    # scarlet is slow, so we use less batches for this example.
    for _ in range(2):
        blend_batch = next(draw_generator)
        scarlet_batch = scarlet(
            blend_batch,
            reference_catalogs=None,  # uses truth catalog
            njobs=1,
        )
        matching_scarlet = matcher(blend_batch.catalog_list, scarlet_batch.catalog_list)

        # need their centroids need to measure ellipticity
        b, ms1, _, _, _ = blend_batch.isolated_images.shape
        centroids1 = np.zeros((b, ms1, 2))
        for jj, t in enumerate(blend_batch.catalog_list):
            n_sources = len(t)
            if n_sources > 0:
                centroids1[jj, :n_sources, 0] = t["x_peak"].value
                centroids1[jj, :n_sources, 1] = t["y_peak"].value

        b, ms2, _, _, _ = scarlet_batch.deblended_images.shape
        centroids2 = np.zeros((b, ms2, 2))
        for kk, t in enumerate(scarlet_batch.catalog_list):
            n_sources = len(t)
            if n_sources > 0:
                centroids2[kk, :n_sources, 0] = t["x_peak"].value
                centroids2[kk, :n_sources, 1] = t["y_peak"].value

        psf_r = blend_batch.psf[2]  # psf in r-band

        true_iso_images = blend_batch.isolated_images[:, :, 2]  # pick 'r' band
        iso_images_scarlet = scarlet_batch.deblended_images[:, :, 2]  # pick 'r' band

        iso_images1, xy1 = matching_scarlet.match_true_arrays(true_iso_images, centroids1)
        iso_images2, xy2 = matching_scarlet.match_pred_arrays(iso_images_scarlet, centroids2)

        ellips1 = get_ksb_ellipticity(iso_images1, xy1, psf_r, pixel_scale=0.2)
        ellips2 = get_ksb_ellipticity(iso_images2, xy2, psf_r, pixel_scale=0.2)

        snr = get_snr(iso_images1, sky_level)
        blendedness = get_blendedness(iso_images1)

        es1.append(ellips1)
        es2.append(ellips2)
        snrs.append(snr)
        bs.append(blendedness)

    e11 = np.concatenate(es1)[:, :, 0].flatten()
    e12 = np.concatenate(es1)[:, :, 1].flatten()
    e21 = np.concatenate(es2)[:, :, 0].flatten()
    e22 = np.concatenate(es2)[:, :, 1].flatten()
    snr = np.concatenate(snrs).flatten()
    bdd = np.concatenate(bs).flatten()

    cond1 = ~np.isnan(e11)
    cond2 = ~np.isnan(e12)
    cond3 = ~np.isnan(e21)
    cond4 = ~np.isnan(e22)
    cond5 = (snr > 0) & (snr < 100)
    cond = cond1 & cond2 & cond3 & cond4 & cond5

    e11 = e11[cond]
    e12 = e12[cond]
    e21 = e21[cond]
    e22 = e22[cond]
    snr = snr[cond]
    bdd = bdd[cond]
