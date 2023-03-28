from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np
from astropy.table import Table
from scipy.optimize import linear_sum_assignment


class Matching(ABC):
    def __init__(self, distance_function: Callable, *args, **kwargs) -> None:
        """Initialize matching class."""
        self.distance_function = distance_function

    @abstractmethod
    def match():
        pass


def distance_center(cat1: Table, cat2: Table) -> np.ndarray:
    """Computes the euclidean distance between the two galaxies given as arguments.

    Args:
        cat1: Catalog containing at least columns `x_peak` and `y_peak`.
        cat2: Catalog containing at least columns `x_peak` and `y_peak`.

    Returns:
        Distance between corresponding galaxies in each catalog as an np.ndarray.
    """
    x_peak1, y_peak1 = cat1["x_peak"].data, cat1["y_peak"].data
    x_peak2, y_peak2 = cat2["x_peak"].data, cat2["y_peak"].data
    return np.hypot(x_peak1 - x_peak2, y_peak1 - y_peak2)


def get_id_matches(truth: Table, pred: Table) -> Tuple[np.ndarray, np.ndarray]:
    """Assume catalogs are already matched one-to-one."""
    dist = distance_center(truth, pred)
    return list(range(len(truth))), dist


def get_least_dist_matches(
    truth: Table,
    pred: Table,
    f_distance: Callable = distance_center,
    dist_thresh: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
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
        truth: Table with entries corresponding to
            the true object parameter values in one blend.
        pred: Table with entries corresponding
            to output of measurement algorithm in one blend.
        f_distance: Function used to compute the distance between true and detected
            galaxies. Takes as arguments the entries corresponding to the two galaxies.
            By default the distance is the euclidean distance from center to center.
        dist_thresh: Maximum distance for matching a detected and a
                true galaxy in pixels.

    Returns:
        Tuple containing two elements:
            - "matches": Index of row in `pred` table corresponding to
                matched detected object. If no match, value is -1.
            - "dist_m": distance between true object and matched object or 0 if not matched.

    """
    if "x_peak" not in truth.colnames:
        raise KeyError("True table has no column x_peak")
    if "y_peak" not in truth.colnames:
        raise KeyError("True table has no column y_peak")
    if "x_peak" not in pred.colnames:
        raise KeyError("Detection table has no column x_peak")
    if "y_peak" not in pred.colnames:
        raise KeyError("Detection table has no column y_peak")

    # dist[i][j] = distance between true object i and detected object j.
    dist = np.zeros((len(truth), len(pred)))
    for i, true_gal in enumerate(truth):
        for j, detected_gal in enumerate(pred):
            dist[i][j] = f_distance(true_gal, detected_gal)

    # solve optimization problem.
    # true_table[true_indx[i]] is matched with detected_table[detected_indx[i]]
    # len(true_indx) = len(detect_indx) = min(len(true_table), len(detected_table))
    true_indx, detected_indx = linear_sum_assignment(dist)

    # for each true galaxy i, match_indx[i] is the index of detected_table matched to that true
    # galaxy or -1 if there is no match.
    match_indx = [-1] * len(truth)
    dist_m = [0.0] * len(truth)
    for i, indx in enumerate(true_indx):
        if dist[indx][detected_indx[i]] <= dist_thresh:
            match_indx[indx] = detected_indx[i]
            dist_m[indx] = dist[indx][detected_indx[i]]

    return match_indx, dist_m
