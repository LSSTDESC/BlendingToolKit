"""Implements a variety of metrics for evaluation results of measurements in BTK."""
import astropy.table
import numpy as np
import skimage.metrics


def get_detection_match(true_table, detected_table):
    """Match detections to true objects.

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        detected_table(astropy.table.Table): Table with entries corresponding
            to output of measurement algorithm in one blend.
    Returns:
        match_table (astropy.table.Table) : Table containing the matches for the
            true galaxies, in the same order as true_table
    """
    match_table = astropy.table.Table()
    if len(detected_table) == 0 or len(true_table) == 0:
        # No match since either no detection or no true objects
        return
    t_x = true_table["x_peak"] - detected_table["x_peak"][:, np.newaxis]
    t_y = true_table["y_peak"] - detected_table["y_peak"][:, np.newaxis]
    dist = np.hypot(t_x, t_y)
    match_table["d_min"] = np.min(dist, axis=0)
    detection_threshold = 5
    condlist = [
        np.min(dist, axis=0) <= detection_threshold,
        np.min(dist, axis=0) > detection_threshold,
    ]
    choicelist = [np.argmin(dist, axis=0), -1]
    match_id = np.select(condlist, choicelist)
    # Not perfect... See how to improve that
    for m in np.unique(match_id[match_id >= 0]):
        idx = np.where(match_id == m)[0]
        best_match = match_table["d_min"][idx].argmin()
        for i in range(len(idx)):
            if i != best_match:
                match_id[idx[i]] = -1
    match_table["match_detected_id"] = match_id
    return match_table


def detection_metrics(blended_images, isolated_images, blend_list, detection_catalogs, matches):
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
        precision.append(true_pos / (true_pos + false_pos))
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
    meas_band_num,
):
    results_segmentation = {}
    iou_results = []
    for i in range(len(blend_list)):
        iou_blend_results = []
        matches_blend = matches[i]["match_detected_id"]
        for j, match in enumerate(matches_blend):
            # TODO : put a correct threshold (according to noise ?)
            threshold = 1.0
            true_segmentation = isolated_images[i][j][meas_band_num] > threshold
            detected_segmentation = segmentations[i][match]
            iou_blend_results.append(
                np.sum(np.logical_and(true_segmentation, detected_segmentation))
                / np.sum(np.logical_or(true_segmentation, detected_segmentation))
            )
        iou_results.append(iou_blend_results)
    results_segmentation["iou"] = iou_results
    return results_segmentation


def reconstruction_metrics(
    blended_images, isolated_images, blend_list, detection_catalogs, deblended_images, matches
):
    results_reconstruction = {}
    mse_results = []
    psnr_results = []
    ssim_results = []
    for i in range(len(blend_list)):
        mse_blend_results = []
        psnr_blend_results = []
        ssim_blend_results = []
        for j in range(len(blend_list[i])):
            match_detected = matches[i]["match_detected_id"][j]
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
        mse_results.append(mse_blend_results)
        psnr_results.append(psnr_blend_results)
        ssim_results.append(ssim_blend_results)
    results_reconstruction["mse"] = mse_results
    results_reconstruction["psnr"] = psnr_results
    results_reconstruction["ssim"] = ssim_results
    return results_reconstruction


def compute_metrics(
    blended_images,
    isolated_images,
    blend_list,
    detection_catalogs,
    segmentations=None,
    deblended_images=None,
    use_metrics=("detection", "segmentation", "reconstruction"),
    meas_band_num=0,
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
        results["segmentation"] = segmentation_metrics(
            blended_images,
            isolated_images,
            blend_list,
            detection_catalogs,
            segmentations,
            matches,
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
        )
    return results


class MetricsGenerator:
    def __init__(self, measure_generator, use_metrics=("detection"), meas_band_num=0):
        self.measure_generator = measure_generator
        self.use_metrics = use_metrics
        self.meas_band_num = meas_band_num

    def __next__(self):
        blend_results, measure_results = next(self.measure_generator)
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
                self.meas_band_num,
            )
            metrics_results[f] = metrics_results_f

        return blend_results, measure_results, metrics_results
