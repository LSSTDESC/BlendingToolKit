"""Implements a variety of metrics for evaluating measurement results in BTK."""
import astropy.table
import galsim
import numpy as np
import skimage.metrics
from scipy.optimize import linear_sum_assignment

from btk.survey import get_mean_sky_level


def meas_ellipticity(image, additional_params):
    psf_image = galsim.Image(image.shape[1], image.shape[2])
    psf_image = additional_params["psf"].drawImage(psf_image)
    pixel_scale = additional_params["pixel_scale"]
    meas_band_num = additional_params["meas_band_num"]
    gal_image = galsim.Image(image[meas_band_num, :, :])
    gal_image.scale = pixel_scale
    shear_est = "KSB"
    try:
        res = galsim.hsm.EstimateShear(gal_image, psf_image, shear_est=shear_est, strict=True)
        result = [res.corrected_g1, res.corrected_g2]
    except RuntimeError as e:
        print(e)
        result = [10.0, 10.0]
    return result


def get_detection_match(true_table, detected_table):
    r"""Uses the Hungarian algorithm to find optimal matching between detections and true objects.
    The optimal matching is computed based on the following optimization problem:
    ```
        \sum_{i} \sum_{j} C_{i,j} X_{i,j}
    ```
    where, in the BTK context, C_{ij}` is the cost function between matching true object `i` with
    detected object `j` computed as the `l2`-distance between the two objects, and `X_{i,j}` is an
    indicator function over the matches.

    Based on this implementation in scipy:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        detected_table(astropy.table.Table): Table with entries corresponding
            to output of measurement algorithm in one blend.

    Returns:
        match_table (astropy.table.Table): Table where each row corresponds to each true
            galaxy in `true_table` containing two columns:
                - "match_detected_id": Index of row in `detected_table` corresponding to
                    matched detected object. If no match, value is -1.
                - "dist": distance between true object and matched object or 0 if no matches.
    """
    match_table = astropy.table.Table()
    t_x = true_table["x_peak"].reshape(-1, 1) - detected_table["x_peak"].reshape(1, -1)
    t_y = true_table["y_peak"].reshape(-1, 1) - detected_table["y_peak"].reshape(1, -1)
    dist = np.hypot(t_x, t_y)  # dist_ij = distance between true object i and detected object j.

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


def detection_metrics(blended_images, isolated_images, blend_list, detection_catalogs, matches):
    """Calculate common detection metrics (f1-score, precision, recall) based on matches.

    NOTE: This function operates directly on batches returned from MeasureGenerator.

    Returns:
        results_detection (dict): Dictionary containing keys "f1", "precision", and "recall".
            Each value is a list where each element corresponds to each element of the batch.
    """
    results_detection = {}
    precision = []
    recall = []
    f1 = []
    for i in range(len(blend_list)):
        matches_blend = matches[i]["match_detected_id"]
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for match in matches_blend:
            if match == -1:
                false_neg += 1
            else:
                true_pos += 1
        for j in range(len(detection_catalogs[i])):
            if j not in matches_blend:
                false_pos += 1
        if true_pos + false_pos > 0:
            precision.append(true_pos / (true_pos + false_pos))
        else:
            precision.append(0)
        recall.append(true_pos / (true_pos + false_neg))
        if precision[-1] != 0 and recall[-1] != 0:
            f1.append(2 / (1 / precision[-1] + 1 / recall[-1]))
        else:
            f1.append(0)
    results_detection["precision"] = precision
    results_detection["recall"] = recall
    results_detection["f1"] = recall
    return results_detection


def segmentation_metrics(
    blended_images,
    isolated_images,
    blend_list,
    detection_catalogs,
    segmentations,
    matches,
    noise_threshold,
    meas_band_num,
):
    results_segmentation = {}
    iou_results = []
    for i in range(len(blend_list)):
        iou_blend_results = []
        matches_blend = matches[i]["match_detected_id"]
        for j, match in enumerate(matches_blend):
            if match != -1:
                true_segmentation = isolated_images[i][j][meas_band_num] > noise_threshold
                detected_segmentation = segmentations[i][match]
                iou_blend_results.append(
                    np.sum(np.logical_and(true_segmentation, detected_segmentation))
                    / np.sum(np.logical_or(true_segmentation, detected_segmentation))
                )
            else:
                iou_blend_results.append(-1)
        iou_results.append(iou_blend_results)
    results_segmentation["iou"] = iou_results
    return results_segmentation


def reconstruction_metrics(
    blended_images,
    isolated_images,
    blend_list,
    detection_catalogs,
    deblended_images,
    matches,
    meas_band_num=0,
    target_meas={},
    psf_images=None,
    pixel_scale=None,
):
    results_reconstruction = {}
    mse_results = []
    psnr_results = []
    ssim_results = []
    target_meas_keys = list(target_meas.keys())
    target_meas_results = []
    additional_params = {
        "psf": psf_images[meas_band_num],
        "pixel_scale": pixel_scale,
        "meas_band_num": meas_band_num,
    }
    for i in range(len(blend_list)):
        mse_blend_results = []
        psnr_blend_results = []
        ssim_blend_results = []
        target_meas_blend_results = {}
        for k in target_meas_keys:
            target_meas_blend_results[k] = []
            target_meas_blend_results[k + "_true"] = []
        for j in range(len(blend_list[i])):
            match_detected = matches[i]["match_detected_id"][j]
            if match_detected != -1:
                mse_blend_results.append(
                    skimage.metrics.mean_squared_error(
                        isolated_images[i][j], deblended_images[i][match_detected]
                    )
                )
                psnr_blend_results.append(
                    skimage.metrics.peak_signal_noise_ratio(
                        isolated_images[i][j],
                        deblended_images[i][match_detected],
                        data_range=np.max(isolated_images[i][j]),
                    )
                )
                ssim_blend_results.append(
                    skimage.metrics.structural_similarity(
                        np.moveaxis(isolated_images[i][j], 0, -1),
                        np.moveaxis(deblended_images[i][match_detected], 0, -1),
                        multichannel=True,
                    )
                )
                for k in target_meas.keys():
                    res_isolated = target_meas[k](isolated_images[i][j], additional_params)
                    res_deblended = target_meas[k](
                        deblended_images[i][match_detected], additional_params
                    )
                    if isinstance(res_isolated, list):
                        if k in target_meas_keys:
                            target_meas_keys.remove(k)
                            target_meas_blend_results.pop(k, None)
                            target_meas_blend_results.pop(k + "_true", None)
                        for res in range(len(res_isolated)):
                            if k + str(res) not in target_meas_keys:
                                target_meas_blend_results[k + str(res)] = []
                                target_meas_blend_results[k + str(res) + "_true"] = []
                                target_meas_keys.append(k + str(res))
                            target_meas_blend_results[k + str(res) + "_true"].append(
                                res_isolated[res]
                            )
                            target_meas_blend_results[k + str(res)].append(res_deblended[res])
                    else:
                        target_meas_blend_results[k + "_true"].append(res_isolated)
                        target_meas_blend_results[k].append(res_deblended)
            else:
                mse_blend_results.append(-1)
                psnr_blend_results.append(-1)
                ssim_blend_results.append(-1)
                for k in target_meas_blend_results.keys():
                    target_meas_blend_results[k].append(-1)
        mse_results.append(mse_blend_results)
        psnr_results.append(psnr_blend_results)
        ssim_results.append(ssim_blend_results)
        target_meas_results.append(target_meas_blend_results)
    results_reconstruction["mse"] = mse_results
    results_reconstruction["psnr"] = psnr_results
    results_reconstruction["ssim"] = ssim_results
    for k in target_meas_keys:
        results_reconstruction[k] = [target_meas_results[i][k] for i in range(len(blend_list))]
        results_reconstruction[k + "_true"] = [
            target_meas_results[i][k + "_true"] for i in range(len(blend_list))
        ]
    return results_reconstruction


def compute_metrics(
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
    psf_images=None,
    pixel_scale=None,
):
    results = {}
    matches = [
        get_detection_match(blend_list[i], detection_catalogs[i]) for i in range(len(blend_list))
    ]
    results["matches"] = matches
    if "detection" in use_metrics:
        results["detection"] = detection_metrics(
            blended_images, isolated_images, blend_list, detection_catalogs, matches
        )
    if "segmentation" in use_metrics:
        if noise_threshold is None:
            raise ValueError("You should provide a noise threshold to get segmentation metrics.")
        results["segmentation"] = segmentation_metrics(
            blended_images,
            isolated_images,
            blend_list,
            detection_catalogs,
            segmentations,
            matches,
            noise_threshold,
            meas_band_num,
        )
    if "reconstruction" in use_metrics:
        results["reconstruction"] = reconstruction_metrics(
            blended_images,
            isolated_images,
            blend_list,
            detection_catalogs,
            deblended_images,
            matches,
            meas_band_num,
            target_meas,
            psf_images,
            pixel_scale,
        )
    names = blend_list[0].colnames
    names += [
        "detected",
        "distance_detection",
        "distance_closest_galaxy",
        "blend_id",
        "blendedness",
    ]
    if "reconstruction" in use_metrics:
        reconstruction_keys = results["reconstruction"].keys()
        names += reconstruction_keys
    if "segmentation" in use_metrics:
        names += ["iou"]
    results["galaxy_summary"] = astropy.table.Table(names=names)
    for i, blend in enumerate(blend_list):
        for j, gal in enumerate(blend):
            row = astropy.table.Table(gal)
            row["detected"] = matches[i]["match_detected_id"][j] != -1
            row["distance_detection"] = matches[i]["dist"][j]
            if len(blend) > 1:
                row["distance_closest_galaxy"] = np.partition(
                    [
                        np.sqrt(
                            (gal["x_peak"] - g["x_peak"]) ** 2 + (gal["y_peak"] - g["y_peak"]) ** 2
                        )
                        for g in blend
                    ],
                    1,
                )[1]
            else:
                row["distance_closest_galaxy"] = 32  # placeholder
            row["blend_id"] = i
            row["blendedness"] = 1 - np.sum(isolated_images[i][j] * isolated_images[i][j]) / np.sum(
                np.sum(isolated_images[i], axis=0) * isolated_images[i][j]
            )
            if "reconstruction" in use_metrics:
                for k in reconstruction_keys:
                    print(results["reconstruction"][k][i])
                    row[k] = results["reconstruction"][k][i][j]
            if "segmentation" in use_metrics:
                row["iou"] = results["segmentation"]["iou"][i][j]
            results["galaxy_summary"].add_row(row[0])

    return results


class MetricsGenerator:
    """Generator that calculates metrics on batches returned by the MeasureGenerator."""

    def __init__(
        self, measure_generator, use_metrics=("detection"), meas_band_num=0, target_meas={}
    ):
        self.measure_generator = measure_generator
        self.use_metrics = use_metrics
        self.meas_band_num = meas_band_num
        self.target_meas = target_meas

    def __next__(self):
        blend_results, measure_results = next(self.measure_generator)
        survey = self.measure_generator.draw_blend_generator.surveys[0]
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
                    self.target_meas,
                    blend_results["psf"],
                    survey.pixel_scale,
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
                self.target_meas,
                blend_results["psf"],
                survey.pixel_scale,
            )

        return blend_results, measure_results, metrics_results
