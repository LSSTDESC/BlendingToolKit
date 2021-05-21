"""Contains utility functions, including functions for loading saved results."""
import os

import numpy as np
from astropy.table import Table

BLEND_RESULT_KEYS = ("blend_images", "isolated_images", "blend_list")


def load_blend_results(path, survey):
    """Load results exported from a DrawBlendsGenerator.

    Args;
        path (str): Path to the files. Should be the same as the save_path
                    which was provided to the DrawBlendsGenerator to save
                    the files.
        survey (str): Name of the survey for which you want to load the files.

    Returns:
        Dictionnary containing the blend images, the isolated images and the
        informations about the blends.
    """
    blend_images = np.load(os.path.join(path, survey, "blended.npy"), allow_pickle=True)
    isolated_images = np.load(os.path.join(path, survey, "isolated.npy"), allow_pickle=True)
    blend_list = [
        Table.read(os.path.join(path, survey, f"blend_info_{i}"), format="ascii")
        for i in range(blend_images.shape[0])
    ]

    return {
        "blend_images": blend_images,
        "isolated_images": isolated_images,
        "blend_list": blend_list,
    }


def load_measure_results(path, measure_name, n_batch):
    """Load results exported from a MeasureGenerator.

    Args:
        path (str): Path to the files. Should be the same as the save_path
                    which was provided to the MeasureGenerator to save
                    the files.
        measure_name (str): Name of the measure function for which you
                    want to load the files
        n_batch (int): Number of blends in the batch you want to load

    Returns:
        Dictionnary containing the detection catalogs, the segmentations
        and the deblended images.
    """
    measure_results = {}
    for key in ["segmentation", "deblended_images"]:
        try:
            measure_results[key] = np.load(
                os.path.join(path, measure_name, f"{key}.npy"), allow_pickle=True
            )
        except FileNotFoundError:
            print(f"No {key} found.")

    catalog = [
        Table.read(
            os.path.join(path, measure_name, f"detection_catalog_{j}"),
            format="ascii",
        )
        for j in range(n_batch)
    ]
    measure_results["catalog"] = catalog
    return measure_results


def load_metrics_results(path, measure_name):
    """Load results exported from a MetricsGenerator.

    Args:
        path (str): Path to the files. Should be the same as the save_path
                    which was provided to the MetricsGenerator to save
                    the files.
        measure_name (str): Name of the measure function for which you
                    want to load the files

    Returns:
        Dictionnary containing the detection catalogs, the segmentations
        and the deblended images.
    """
    metrics_results = {}
    for key in ["detection", "segmentation", "reconstruction"]:
        try:
            metrics_results[key] = np.load(
                os.path.join(path, measure_name, f"{key}_metric.npy"), allow_pickle=True
            )
        except FileNotFoundError:
            print(f"No {key} metrics found.")

    metrics_results["galaxy_summary"] = Table.read(
        os.path.join(path, measure_name, "galaxy_summary"),
        format="ascii",
    )
    return metrics_results


def load_all_results(path, surveys, measure_names, n_batch, n_meas_kwargs=1):
    """Load results exported from a MetricsGenerator.

    Args:
        path (str): Path to the files. Should be the same as the save_path
                    which was provided to the MetricsGenerator to save
                    the files.
        surveys (list): Names of the surveys for which you want to load
                        the files
        measure_names (list): Names of the measure functions for which you
                    want to load the files
        n_batch (int): Number of blends in the batch you want to load

    Returns:
        The three dictionnaries corresponding to the results.
    """
    blend_results = {}
    for key in BLEND_RESULT_KEYS:
        blend_results[key] = {}
    measure_results = {}
    metrics_results = {}
    for s in surveys:
        blend_results_temp = load_blend_results(path, s)
        for key in BLEND_RESULT_KEYS:
            blend_results[key][s] = blend_results_temp[key]

    measure_results = {"catalog": {}, "segmentation": {}, "deblended_images": {}}
    for meas in measure_names:
        for n in range(n_meas_kwargs):
            dir_name = meas + str(n) if n_meas_kwargs > 1 else meas
            meas_results = load_measure_results(path, dir_name, n_batch)
            for k in meas_results.keys():
                measure_results[k][dir_name] = meas_results[k]
            metrics_results[dir_name] = load_metrics_results(path, dir_name)

    return blend_results, measure_results, metrics_results
