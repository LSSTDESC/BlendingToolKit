"""Implements a variety of metrics for evaluating measurement results in BTK.

BTK users are expected to use the MetricsGenerator class, which is initialized by providing
a MeasureGenerator as well as some parameters. Users which do not want to use the full BTK
pipeline may use the compute_metrics function which takes the raw data as an input.

Currently, we support the following metrics :

* For detection, all metrics are per batch :

  * Number of true positives, ie number of true galaxies which have been correctly detected
  * Number of false positives, ie number of detected galaxies which do not correspond
    to a true galaxy
  * Number of false negatives, ie number of true galaxies which have not been detected
  * Precision, the ratio of true positives against the total number of positives ; describes
    how much the algorithm is susceptible to make false detections (closer to 1 is better)
  * Recall, the ratio of true positives against the number of true galaxies (which is equal
    to true positives + false negatives) ; indicates the capacity of the algorithm for
    detecting all the galaxies (closer to 1 is better)
  * F1 score, the harmonic mean of precision and recall ; gives an overall assessment of the
    detection (closer to 1 is better)
  * Efficiency matrix, contains for each possible number of true galaxies in a blend the
    distribution of the number of detected galaxies in blends containing this number of true
    galaxies.

* For segmentation, all metrics are per galaxy :

  * Intersection-over-Union (IoU), the ratio between the intersection of the true and
    detected segmentations (true segmentation is computed by applying a threshold on
    the true image) and the union of the two. Closer to 1 is better.

* For reconstruction, all metrics are per galaxy :

  * Mean Square Residual (MSR), the mean square error between the true image and the
    deblended image. Lower is better.
  * Peak Signal to Noise Ratio (PSNR), the log of the maximum value in the image divided
    by the MSR squared (result is in dB). Higher is better.
  * Structure Similarity Index (SSIM), a more advanced metric based on perceptual
    considerations, divided in luminance, contrast and structure. Closer to 1 is better.

* Additionnal information provided :

  * Distance between the detection and the true galaxy
  * Distance to the closest true galaxy
  * Blendedness, defined as :

    .. math::

        1 - \\frac{S_k \\cdot S_k}{S_{all} \\cdot S_k}

    where :math:`S_k` is the flux of the k-th galaxy for each pixel (as a vector),
    :math:`S_{all}` is the flux of all the galaxies for each pixel, and :math:`\\cdot`
    is the standard scalar product on vectors.

"""
import astropy.table
import galsim
import numpy as np
import skimage.metrics
from scipy.optimize import linear_sum_assignment

from btk.measure import MeasureGenerator
from btk.survey import get_mean_sky_level


def get_blendedness(iso_image, blend_iso_images):
    """Calculate blendedness given isolated images of each galaxy in a blend.

    Args:
        iso_image (np.array): Array of shape = (H,W) corresponding to image of the isolated
            galaxy you are calculating blendedness for.
        blend_iso_images (np.array): Array of shape = (N, H, W) where N is the number of galaxies
            in the blend and each image of this array corresponds to an isolated galaxy that is
            part of the blend (includes `iso_image`).
    """
    num = np.sum(iso_image * iso_image)
    denom = np.sum(np.sum(blend_iso_images, axis=0) * iso_image)
    return 1 - num / denom


def meas_ksb_ellipticity(image, additional_params):
    """Utility function to measure ellipticity using the `galsim.hsm` package, with
    the KSB method.

    Args:
        image (np.array): Image of a single, isolated galaxy with shape (H, W).
        additional_params (dict): Containing keys 'psf', 'pixel_scale' and 'meas_band_num'.
                                  The psf should be a Galsim PSF model, and meas_band_num
                                  an integer indicating the band in which the measurement
                                  is done.
    """
    psf_image = galsim.Image(image.shape[1], image.shape[2])
    psf_image = additional_params["psf"].drawImage(psf_image)
    pixel_scale = additional_params["pixel_scale"]
    meas_band_num = additional_params["meas_band_num"]
    gal_image = galsim.Image(image[meas_band_num, :, :])
    gal_image.scale = pixel_scale
    shear_est = "KSB"
    try:
        res = galsim.hsm.EstimateShear(gal_image, psf_image, shear_est=shear_est, strict=True)
        result = [res.corrected_g1, res.corrected_g2, res.observed_shape.e]
    except RuntimeError as e:
        print(e)
        result = [10.0, 10.0, 10.0]
    return result


def get_detection_match(true_table, detected_table):
    r"""Uses the Hungarian algorithm to find optimal matching between detections and true objects.

    The optimal matching is computed based on the following optimization problem:

    .. math::

        \sum_{i} \sum_{j} C_{i,j} X_{i,j}

    where, in the BTK context, :math:`C_{ij}` is the cost function between matching true object
    :math:`i` with detected object :math:`j` computed as the L2 distance between the two objects,
    and :math:`X_{i,j}` is an indicator function over the matches.

    Based on this implementation in scipy:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        detected_table(astropy.table.Table): Table with entries corresponding
            to output of measurement algorithm in one blend.

    Returns:
        match_table (astropy.table.Table): Table where each row corresponds to each true
        galaxy in `true_table` and contains two columns:

        - "match_detected_id": Index of row in `detected_table` corresponding to
          matched detected object. If no match, value is -1.
        - "dist": distance between true object and matched object or 0 if no matches.

    """
    match_table = astropy.table.Table()
    t_x = true_table["x_peak"].reshape(-1, 1) - detected_table["x_peak"].reshape(1, -1)
    t_y = true_table["y_peak"].reshape(-1, 1) - detected_table["y_peak"].reshape(1, -1)
    dist = np.hypot(t_x, t_y)  # dist[i][j] = distance between true object i and detected object j.

    # solve optimization problem.
    # true_table[true_indx[i]] is matched with detected_table[detected_indx[i]]
    # len(true_indx) = len(detect_indx) = min(len(true_table), len(detected_table))
    true_indx, detected_indx = linear_sum_assignment(dist)

    # for each true galaxy i, match_indx[i] is the index of detected_table matched to that true
    # galaxy or -1 if there is no match.
    match_indx = [-1] * len(true_table)
    dist_m = [0.0] * len(true_table)
    for i, indx in enumerate(true_indx):
        match_indx[indx] = detected_indx[i]
        dist_m[indx] = dist[indx][detected_indx[i]]

    match_table["match_detected_id"] = match_indx
    match_table["dist"] = dist_m
    return match_table


def get_detection_eff_matrix(summary_table, num):
    """Computes the detection efficiency matrix for the input detection summary
    table.
    Input argument num sets the maximum number of true objects per blend in the
    test set for which the
    detection efficiency matrix is to be created for. Detection efficiency is
    computed for a number of true objects in the range (0-num) as columns and
    the detection percentage as rows. The percentage values in a column sum to
    100.
    The input summary table must be a numpy array of shape [N, 5], where N is
    the test set size. The 5 columns in the summary_table are number of true
    objects, detected sources, undetected objects, spurious detections and
    shredded objects for each of the N blend scenes in the test set.

    Args:
        summary_table (`numpy.array`): Detection summary as a table [N, 5].
        num (int): Maximum number of true objects to create matrix for. Number
                   of columns in efficiency matrix will be num+1. The first column
                   will correspond to no true objects.
    Returns:
        numpy.ndarray of size[num+2, num+1] that shows detection efficiency.
    """
    eff_matrix = np.zeros((num + 2, num + 1))
    for i in range(0, num + 1):
        (q_true,) = np.where(summary_table[:, 0] == i)
        for j in range(0, num + 2):
            if len(q_true) > 0:
                (q_det,) = np.where(summary_table[q_true, 1] == j)
                eff_matrix[j, i] = len(q_det)
    norm = np.sum(eff_matrix, axis=0)
    # If no detections along a column, set sum to 1 to avoid dividing by zero.
    norm[norm == 0.0] = 1
    # normalize over columns.
    eff_matrix = eff_matrix / norm[np.newaxis, :] * 100.0
    return eff_matrix


def detection_metrics(detection_catalogs, matches):
    """Calculate detection metrics based on matches from `get_detection_match` function.

    Currently implemented detection metrics include:
    - recall
    - precision
    - f1

    Args:
        detection_catalogs (list) : Contains one astropy Table for each blend,
                                    corresponding to the detected galaxies.
        matches (list) : Contains one astropy Table for each blend, with a
                         column `match_detected_id` containing the index of the
                         matched detected galaxy for each true galaxy.
    Returns:
        results_detection (dict): Dictionary containing keys corresponding to each implemented
        metric. Each value is a list where each element corresponds to each element of
        the batch (a single blend).
    """
    results_detection = {}
    true_pos = 0
    false_pos = 0
    false_neg = 0
    efficiency_input_table = []
    for i in range(len(matches)):
        matches_blend = matches[i]["match_detected_id"]
        for match in matches_blend:
            if match == -1:
                false_neg += 1
            else:
                true_pos += 1
        for j in range(len(detection_catalogs[i])):
            if j not in matches_blend:
                false_pos += 1
        efficiency_input_table.append([len(matches[i]), len(detection_catalogs[i])])
    results_detection["true_pos"] = true_pos
    results_detection["false_pos"] = false_pos
    results_detection["false_neg"] = false_neg
    results_detection["precision"] = true_pos / (true_pos + false_pos)
    results_detection["recall"] = true_pos / (true_pos + false_neg)
    results_detection["f1"] = 2 / (
        1 / results_detection["precision"] + 1 / results_detection["recall"]
    )
    results_detection["eff_matrix"] = get_detection_eff_matrix(
        np.array(efficiency_input_table), np.max([len(match) for match in matches])
    )
    return results_detection


def segmentation_metrics_blend(
    isolated_images, detected_segmentations, matches, noise_threshold, meas_band_num
):
    """Calculates segmentation metrics given information from a single blend.
    The true segmentation is obtained from the isolated images by setting to True
    the pixels above the noise_threshold in the band meas_band_num.

    Args:
        isolated_images (array) : Contains the isolated true galaxy images,
                                  with shape MCHW where M is the number of
                                  galaxies in the blend.
        detected_segmentations (array) : Contains the detected segmentations,
                                         as a boolean array with shape LHW, where
                                         L is the number of detected galaxies.
        matches (astropy.table.Table) : Contains the index of the matching detected
                                        galaxy for each true galaxy, under the column
                                        `match_detected_id`.
        noise_threshold (float) : Threshold for a pixel to be considered belonging to
                                  an object in the segmentation. Should be based on the
                                  expected noise level in the full image.
        meas_band_num (int) : Index of the band in which the true segmentation is computed.

    Returns:
        iou_blend_result (list) : Contains the results for the IoU metric for each
        galaxy.

    """
    iou_blend_results = []
    matches_blend = matches["match_detected_id"]
    for j, match in enumerate(matches_blend):
        if match != -1:
            true_segmentation = isolated_images[j][meas_band_num] > noise_threshold
            detected_segmentation = detected_segmentations[match]
            iou_blend_results.append(
                np.sum(np.logical_and(true_segmentation, detected_segmentation))
                / np.sum(np.logical_or(true_segmentation, detected_segmentation))
            )
        else:
            iou_blend_results.append(-1)
    return iou_blend_results


def segmentation_metrics(
    isolated_images,
    segmentations,
    matches,
    noise_threshold,
    meas_band_num,
):
    """Calculates segmentation metrics given information from a single batch.

    Currently implemented segmentation metrics include:
    - Intersection-over-Union (IOU)

    Args:
        isolated_images (array) : Contains the isolated true galaxy images,
                                  with shape NMCHW where M is the maximum number
                                  of galaxies in the blend and N the number of
                                  blends in a batch.
        segmentations (array) : Contains the detected segmentations,
                                as a boolean array with shape NLHW, where
                                L is the number of detected galaxies and N
                                the number of blends in a batch.
        matches (list) : Contains one astropy Table for each blend, with a
                         column `match_detected_id` containing the index of the
                         matched detected galaxy for each true galaxy.
        noise_threshold (float) : Threshold for a pixel to be considered belonging to
                                  an object in the segmentation. Should be based on the
                                  expected noise level in the full image.
        meas_band_num (int) : Index of the band in which the true segmentation is computed.

    Returns:
        results_segmentation (astropy.table.Table) : Contains the results for the batch
        with one column for each metric (currently : "iou" for IoU).
    """
    results_segmentation = {}
    iou_results = []
    for i in range(len(isolated_images)):
        iou_blend_results = segmentation_metrics_blend(
            isolated_images[i], segmentations[i], matches[i], noise_threshold, meas_band_num
        )
        iou_results.append(iou_blend_results)
    results_segmentation["iou"] = iou_results
    return results_segmentation


def reconstruction_metrics_blend(
    isolated_images, deblended_images, matches, target_meas, target_meas_keys
):
    """Calculates reconstruction metrics given information from a single blend.

    Args:
        isolated_images (array) : Contains the isolated true galaxy images,
                                  with shape MCHW where M is the number of
                                  galaxies in the blend.
        deblended_images (array) : Contains the deblended images,
                                   as a boolean array with shape LCHW, where
                                   L is the number of detected galaxies.
        matches (astropy.table.Table) : Contains the index of the matching detected
                                        galaxy for each true galaxy, under the column
                                        `match_detected_id`.
        target_meas (dict) : Contains the target measurement functions provided
                             by the user.
        target_meas_key (list) : Contains the relevant keys, to manage the case
                                 where one of the target measurement functions
                                 has multiple outputs.
    Returns:
        msr_blend_result (list) : Contains the results for the Mean Square Residual
                                  metric for each galaxy.
        psnr_blend_result (list) : Contains the results for the Peak Signal to Noise
                                   Ratio metric for each galaxy.
        ssim_blend_result (list) : Contains the results for the Structure Similarity
                                   Index metric for each galaxy.
        target_meas_blend_result (dict) : Contains the results for the target
                                          measurement function for each galaxy (both
                                          true and deblended images).
    """

    msr_blend_results = []
    psnr_blend_results = []
    ssim_blend_results = []
    target_meas_blend_results = {}
    for k in target_meas_keys:
        target_meas_blend_results[k] = []
        target_meas_blend_results[k + "_true"] = []
    for j in range(len(matches["match_detected_id"])):
        match_detected = matches["match_detected_id"][j]
        if match_detected != -1:
            msr_blend_results.append(
                skimage.metrics.mean_squared_error(
                    isolated_images[j], deblended_images[match_detected]
                )
            )
            psnr_blend_results.append(
                skimage.metrics.peak_signal_noise_ratio(
                    isolated_images[j],
                    deblended_images[match_detected],
                    data_range=np.max(isolated_images[j]),
                )
            )
            ssim_blend_results.append(
                skimage.metrics.structural_similarity(
                    np.moveaxis(isolated_images[j], 0, -1),
                    np.moveaxis(deblended_images[match_detected], 0, -1),
                    multichannel=True,
                )
            )
            for k in target_meas.keys():
                res_deblended = target_meas[k](deblended_images[match_detected])
                res_isolated = target_meas[k](isolated_images[j])
                if isinstance(res_isolated, list):
                    for res in range(len(res_isolated)):
                        target_meas_blend_results[k + str(res)].append(res_deblended[res])
                        target_meas_blend_results[k + str(res) + "_true"].append(res_isolated[res])
                else:
                    target_meas_blend_results[k].append(res_deblended)
                    target_meas_blend_results[k + "_true"].append(res_isolated)
        else:
            msr_blend_results.append(-1)
            psnr_blend_results.append(-1)
            ssim_blend_results.append(-1)
            for k in target_meas_blend_results.keys():
                target_meas_blend_results[k].append(-1)

    return msr_blend_results, psnr_blend_results, ssim_blend_results, target_meas_blend_results


def reconstruction_metrics(
    isolated_images,
    deblended_images,
    matches,
    target_meas={},
):
    """Calculate reconstruction metrics given information from a single batch.

    Currently implemented reconstruction metrics include:
    - Mean Squared Residual (MSR)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Structural Similarity (SSIM)

    Args:
        isolated_images (array) : Contains the isolated true galaxy images,
                                  with shape NMCHW where M is the maximum number
                                  of galaxies in the blend and N the number of
                                  blends in a batch.
        deblended_images (array) : Contains the deblended images,
                                   as an array with shape NLCHW, where
                                   L is the number of detected galaxies and N
                                   the number of blends in a batch.
        matches (list) : Contains one astropy Table for each blend, with a
                         column `match_detected_id` containing the index of the
                         matched detected galaxy for each true galaxy.
        target_meas (dict) : Contains the target measurement functions provided
                             by the user.

    Returns:
        results_reconstruction (astropy.table.Table) : Contains the results for the batch
       with one column for each metric (currently : "msr"
       for MSR, "psnr" for PSNR, "ssim" for SSIM, and
       "target_name" and "target_name_true" for each
       function target_name in target_meas (if the
       function has several outputs, a number is added
       after target_name for each output)).
    """
    results_reconstruction = {}
    msr_results = []
    psnr_results = []
    ssim_results = []
    target_meas_keys = list(target_meas.keys())
    for k in target_meas.keys():
        res_0 = target_meas[k](isolated_images[0][0])
        if isinstance(res_0, list):
            target_meas_keys.remove(k)
            for j in range(len(res_0)):
                target_meas_keys.append(k + str(j))

    target_meas_results = []

    for i in range(len(isolated_images)):
        (
            msr_blend_results,
            psnr_blend_results,
            ssim_blend_results,
            target_meas_blend_results,
        ) = reconstruction_metrics_blend(
            isolated_images[i], deblended_images[i], matches[i], target_meas, target_meas_keys
        )
        msr_results.append(msr_blend_results)
        psnr_results.append(psnr_blend_results)
        ssim_results.append(ssim_blend_results)
        target_meas_results.append(target_meas_blend_results)
    results_reconstruction["msr"] = msr_results
    results_reconstruction["psnr"] = psnr_results
    results_reconstruction["ssim"] = ssim_results
    for k in target_meas_keys:
        results_reconstruction[k] = [target_meas_results[i][k] for i in range(len(isolated_images))]
        results_reconstruction[k + "_true"] = [
            target_meas_results[i][k + "_true"] for i in range(len(isolated_images))
        ]
    return results_reconstruction


def compute_metrics(  # noqa: C901
    blended_images,
    isolated_images,
    blend_list,
    detection_catalogs,
    segmentations=None,
    deblended_images=None,
    use_metrics=("detection", "segmentation", "reconstruction"),
    noise_threshold=None,
    meas_band_num=0,
    target_meas={},
    blend_id_start=0,
    channels_last=False,
):
    """Computes all requested metrics given information in a single batch from measure_generator.

    Args:
        blended_images (array) : Contains all the blend images, with shape as specified
                                 by channels_last.
        isolated_images (array) : Contains all the isolated images, with shape NMCHW OR NMHWC
                                  depending on channels_last, with M the maximum number of galaxies
                                  in a blend.
        blend_list (list) : Contains the information related to all blends, as a list of astropy
                            Tables (one for each blend). Those tables should at least
                            contain columns indicating the position in pixels of each galaxy,
                            named "x_peak" and "y_peak".
        detection_catalogs (list) : Contains the information on the detections for all blends, as a
                                    list of astropy Tables (one for each blend). Those tables
                                    should at least contain columns indicating the position
                                    in pixels of each detected galaxy, named "x_peak" and "y_peak".
        segmentations (list) : Contains the measured segmentations, as a list of boolean arrays of
                               shape MHW where M is the number of detected objects (must be
                               consistent with corresponding detection catalog).
        deblended_images (list) : Contains the deblended images, as a list of arrays of shape NCHW
                                or NHWC depending on channels_last, where N is the number of
                                detected objects (must be consistent with corresponding
                                detection catalogs).
        use_metrics (tuple) : Specifies which metrics are to be computed ; can contain "detection",
                              "segmentation" and "reconstruction".
        noise_threshold (float) : Threshold to use when computing the true segmentations from
                                  isolated images.
        meas_band_num (int) : Indicates in which band some of the measurements should be done.
        target_meas (dict) : Contains functions measuring target parameters on images, which will
                             be returned for both isolated and deblended images to compare.
        blend_id_start (int): At what index to start counting each blend.
        channels_last (bool) : Indicates whether the images should be channels first (NCHW)
                          or channels last (NHWC).

    Returns:
        results (dict) : Contains all the computed metrics. Entries are :
                         - matches : list of astropy Tables containing the matched detected galaxy
                           for each true galaxy
                         - detection : dict containing the raw results for detection
                         - segmentation : dict containing the raw results for segmentation
                         - reconstruction : dict containing the raw results for reconstruction
                         - galaxy_summary : astropy Table containing all the galaxies from all
                           blends and related metrics
    """
    if channels_last:
        blended_images = np.moveaxis(blended_images, -1, 1)
        isolated_images = np.moveaxis(isolated_images, -1, 2)
        if deblended_images is not None:
            deblended_images = [np.moveaxis(im, -1, 1) for im in deblended_images]
    results = {}
    matches = [
        get_detection_match(blend_list[i], detection_catalogs[i]) for i in range(len(blend_list))
    ]
    results["matches"] = matches

    names = blend_list[0].colnames
    names += [
        "detected",
        "distance_detection",
        "distance_closest_galaxy",
        "blend_id",
        "blendedness",
    ]

    if "detection" in use_metrics:
        results["detection"] = detection_metrics(detection_catalogs, matches)
    if "segmentation" in use_metrics:
        if noise_threshold is None:
            raise ValueError("You should provide a noise threshold to get segmentation metrics.")
        if segmentations is None:
            raise ValueError("You should provide segmentations to get segmentation metrics")
        results["segmentation"] = segmentation_metrics(
            isolated_images,
            segmentations,
            matches,
            noise_threshold,
            meas_band_num,
        )
        names += ["iou"]
    if "reconstruction" in use_metrics:
        if deblended_images is None:
            raise ValueError("You should provide deblended images to get reconstruction metrics")
        results["reconstruction"] = reconstruction_metrics(
            isolated_images,
            deblended_images,
            matches,
            target_meas,
        )
        reconstruction_keys = results["reconstruction"].keys()
        names += reconstruction_keys

    results["galaxy_summary"] = astropy.table.Table(names=names)
    for i, blend in enumerate(blend_list):
        for j, gal in enumerate(blend):
            row = astropy.table.Table(gal)
            row["detected"] = matches[i]["match_detected_id"][j] != -1
            row["distance_detection"] = matches[i]["dist"][j]

            # obtain distance to closest galaxy in the blend
            if len(blend) > 1:
                dists = []
                for g in blend:
                    dx = gal["x_peak"] - g["x_peak"]
                    dy = gal["y_peak"] - g["y_peak"]
                    dists.append(np.hypot(dx, dy))
                row["distance_closest_galaxy"] = np.partition(dists, 1)[1]
            else:
                row["distance_closest_galaxy"] = -1  # placeholder

            row["blend_id"] = i + blend_id_start
            row["blendedness"] = get_blendedness(isolated_images[i][j], isolated_images[i])
            if "segmentation" in use_metrics:
                row["iou"] = results["segmentation"]["iou"][i][j]
            if "reconstruction" in use_metrics:
                for k in reconstruction_keys:
                    row[k] = results["reconstruction"][k][i][j]
            results["galaxy_summary"].add_row(row[0])

    return results


class MetricsGenerator:
    """Generator that calculates metrics on batches returned by the MeasureGenerator."""

    def __init__(
        self,
        measure_generator,
        use_metrics=("detection"),
        meas_band_num=0,
        target_meas={},
    ):
        """Initialize metrics generator.

        Args:
            measure_generator (btk.measure.MeasureGenerator): Measurement generator object.
            use_metrics (tuple): Which metrics do you want to use? Options:
                                - "detection"
                                - "segmentation"
                                - "reconstruction"
            meas_band_num (int): If using multiple bands for each blend,
                which band index do you want to use for measurement?
            target_meas (dict): Dictionary containing functions that can measure a physical
                parameter on isolated galaxy images. Each key is the name of the estimator and
                value the function performing the estimation (e.g. `meas_ellipticity` above).
        """
        self.measure_generator: MeasureGenerator = measure_generator
        self.use_metrics = use_metrics
        self.meas_band_num = meas_band_num
        self.target_meas = target_meas
        self.blend_counter = 0

    def __next__(self):
        """Returns metric results calculated on one batch."""
        blend_results, measure_results = next(self.measure_generator)
        survey = self.measure_generator.draw_blend_generator.surveys[0]
        additional_params = {
            "psf": blend_results["psf"][self.meas_band_num],
            "pixel_scale": survey.pixel_scale,
            "meas_band_num": self.meas_band_num,
        }
        target_meas = {}
        for k in self.target_meas.keys():
            target_meas[k] = lambda x: self.target_meas[k](x, additional_params)

        noise_threshold = get_mean_sky_level(survey, survey.filters[self.meas_band_num])
        if "catalog" not in measure_results.keys():
            meas_func = measure_results.keys()
            metrics_results = {}
            for f in meas_func:
                metrics_results_f = compute_metrics(
                    blend_results["blend_images"],
                    blend_results["isolated_images"],
                    blend_results["blend_list"],
                    measure_results[f]["catalog"],
                    measure_results[f]["segmentation"],
                    measure_results[f]["deblended_images"],
                    self.use_metrics,
                    noise_threshold,
                    self.meas_band_num,
                    target_meas,
                    blend_id_start=self.blend_counter,
                    channels_last=self.measure_generator.channels_last,
                )
                metrics_results[f] = metrics_results_f

        else:
            metrics_results = compute_metrics(
                blend_results["blend_images"],
                blend_results["isolated_images"],
                blend_results["blend_list"],
                measure_results["catalog"],
                measure_results["segmentation"],
                measure_results["deblended_images"],
                self.use_metrics,
                noise_threshold,
                self.meas_band_num,
                target_meas,
                blend_id_start=self.blend_counter,
                channels_last=self.measure_generator.channels_last,
            )

        self.blend_counter += len(blend_results["blend_list"])
        return blend_results, measure_results, metrics_results


def run_metrics(metrics_generator: MetricsGenerator, n_batches=100):
    """Uses a `metrics_generator` objec to summarize metrics results for `n_batches` batches."""
    measure_funcs = metrics_generator.measure_generator.measure_functions
    summary_tables = {f: astropy.table.Table() for f in measure_funcs}
    for i in range(n_batches):
        blend_results, measure_results, metrics_results = next(metrics_generator)
        for f in measure_funcs:
            summary_tables[f] = astropy.table.vstack(
                summary_tables[f], metrics_results["galaxy_summary"]
            )
    return summary_tables
