"""Implements a variety of metrics for evaluation results of measurements in BTK."""
import astropy.table
import numpy as np
import skimage.metrics
from scipy.optimize import linear_sum_assignment


def get_detection_match_new(true_table, detected_table):
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
        match_table (astropy.table.Table) : Table containing the matches for the
            true galaxies, in the same order as true_table
    """
    match_table = astropy.table.Table()
    if len(detected_table) == 0 or len(true_table) == 0:
        # No match since either no detection or no true objects
        return
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
    for i, indx in enumerate(true_indx):
        match_indx[indx] = detected_indx[i]

    match_table["match_detected_id"] = match_indx
    return match_table


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
        return None
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
        if matches[i] is not None:
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
        else:
            precision.append(0)
            recall.append(0)
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
        if matches[i] is not None:
            iou_blend_results = []
            matches_blend = matches[i]["match_detected_id"]
            for j, match in enumerate(matches_blend):
                # TODO : put a correct threshold (according to noise ?)
                if match != -1:
                    threshold = 1.0
                    true_segmentation = isolated_images[i][j][meas_band_num] > threshold
                    detected_segmentation = segmentations[i][match]
                    iou_blend_results.append(
                        np.sum(np.logical_and(true_segmentation, detected_segmentation))
                        / np.sum(np.logical_or(true_segmentation, detected_segmentation))
                    )
                else:
                    iou_blend_results.append(-1)
            iou_results.append(iou_blend_results)
        else:
            iou_results.append([-1 for j in range(len(blend_list[i]))])
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
        if matches[i] is not None:
            mse_blend_results = []
            psnr_blend_results = []
            ssim_blend_results = []
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
                else:
                    mse_blend_results.append(-1)
                    psnr_blend_results.append(-1)
                    ssim_blend_results.append(-1)
            mse_results.append(mse_blend_results)
            psnr_results.append(psnr_blend_results)
            ssim_results.append(ssim_blend_results)
        else:
            mse_results.append([-1 for j in range(len(blend_list[i]))])
            psnr_results.append([-1 for j in range(len(blend_list[i]))])
            ssim_results.append([-1 for j in range(len(blend_list[i]))])
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
    names = blend_list[0].colnames
    if "reconstruction" in use_metrics:
        names += ["mse", "psnr", "ssim"]
    if "segmentation" in use_metrics:
        names += ["iou"]
    results["galaxy_summary"] = astropy.table.Table(names=names)
    for i, blend in enumerate(blend_list):
        for j, gal in enumerate(blend):
            row = astropy.table.Table(gal)
            if "reconstruction" in use_metrics:
                row.add_columns([[0.0], [0.0], [0.0]], names=["mse", "psnr", "ssim"])
                row["mse"] = results["reconstruction"]["mse"][i][j]
                row["psnr"] = results["reconstruction"]["psnr"][i][j]
                row["ssim"] = results["reconstruction"]["ssim"][i][j]
            if "segmentation" in use_metrics:
                row.add_column([0.0], name="iou")
                row["iou"] = results["segmentation"]["iou"][i][j]
            results["galaxy_summary"].add_row(row[0])

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
