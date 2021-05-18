"""Contains utility functions, including functions for loading saved results."""
import numpy as np
from astropy.table import Table


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
    blend_images = np.load(f"{path}_{survey}_blended.npy", allow_pickle=True)
    isolated_images = np.load(f"{path}_{survey}_isolated.npy", allow_pickle=True)
    blend_list = [Table.read(f"{path}_{survey}_blend_info_{i}", format="ascii")
                          for i in range(blend_images.shape[0]]
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
            measure_results[key] = np.load(f"{path}_{measure_name}_{key}.npy", allow_pickle=True)
        except FileNotFoundError:
            print(f"No {key} found.")
    catalog = [Table.read(f"{path}_{measure_name}_detection_catalog_{j}", format="ascii")
                       for j in range(n_batch)]
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
                f"{path}_{measure_name}_{key}_metric.npy", allow_pickle=True
            )
        except FileNotFoundError:
            print(f"No {key} metrics found.")

    metrics_results["galaxy_summary"] = Table.read(
        f"{path}_{measure_name}_galaxy_summary",
        format="ascii",
    )
    return metrics_results


def load_all_results(path, surveys, measure_names, n_batch):
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
    for key in ["blend_images", "isolated_images", "blend_list"]:
        blend_results[key] = {}
    measure_results = {}
    metrics_results = {}
    for s in surveys:
        blend_results_temp = load_blend_results(path, s)
        for key in ["blend_images", "isolated_images", "blend_list"]:
            blend_results[key][s] = blend_results_temp[key]

    for meas in measure_names:
        measure_results[meas] = load_measure_results(path, meas, n_batch)
        metrics_results[meas] = load_metrics_results(path, meas)

    return blend_results, measure_results, metrics_results
