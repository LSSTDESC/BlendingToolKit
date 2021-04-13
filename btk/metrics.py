"""Implements a variety of metrics for evaluation results of measurements in BTK."""
import astropy.table
import numpy as np
import skimage.metrics


def get_detection_match(true_table, detected_table):
    """Match detections to true objects.

    Function does not return anything, only the astropy tables are updated.

    Args:
        true_table (astropy.table.Table): Table with entries corresponding to
            the true object parameter values in one blend.
        detected_table(astropy.table.Table): Table with entries corresponding
            to output of measurement algorithm in one blend.
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
    match_table["match_detected_id"] = match_id
    return match_table


def detection_metrics(blended_images, isolated_images, blend_list, detection_catalogs, matches):
    results_detection = {}
    precision = []
    recall = []
    for i in range(len(blend_list)):
        matches_blend = matches[i]
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for j in matches_blend:
            pass
        precision.append(true_pos / (true_pos + false_pos))
        recall.append(true_pos / (true_pos + false_neg))
    results_detection["precision"] = precision
    results_detection["recall"] = recall
    return results_detection


def segmentation_metrics(
    blended_images, isolated_images, blend_list, detection_catalogs, segmentations, matches
):
    return 0


def reconstruction_metrics(
    blended_images, isolated_images, blend_list, detection_catalogs, deblended_images, matches
):
    results_reconstruction = {}
    mse_results = []
    for i in range(len(blend_list)):
        mse_blend_results = []
        for j in range(len(blend_list[i])):
            match_detected = matches[i]["match_detected_id"][j]
            mse_blend_results.append(
                skimage.metrics.mean_squared_error(
                    isolated_images[i][j], deblended_images[i][match_detected]
                )
            )
        mse_results.append(mse_blend_results)
    results_reconstruction["mse"] = mse_results
    return results_reconstruction


def compute_metrics(
    blended_images,
    isolated_images,
    blend_list,
    detection_catalogs=None,
    segmentations=None,
    deblended_images=None,
    use_metrics=("detection", "segmentation", "reconstruction"),
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
            blended_images, isolated_images, blend_list, detection_catalogs, segmentations, matches
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


def compute_metrics_wrap(
    blend_results, measure_results, use_metrics=("detection", "segmentation", "reconstruction")
):
    return compute_metrics(
        blend_results["blend_images"],
        blend_results["isolated_images"],
        blend_results["blend_list"],
        measure_results["catalog"],
        measure_results["segmentation"],
        measure_results["deblended_images"],
        use_metrics,
    )
