from abc import ABC, abstractmethod
import numpy as np
from astropy import units
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.optimize import linear_sum_assignment
from scipy import spatial
from typing import Tuple

def pixel_l2_distance_matrix(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> np.ndarray:
    """Computes the pixel L2 (Ecludian) distance matrix between target objects (x1, y1)
    and detected objects (x2, y2).

    Args:
        x1: an array of x-coordinate(s) of target objects in pixels
        y1: an array of y-coordinate(s) of target objects in pixels
        x2: an array of x-coordinate(s) of detected objects in pixels
        y2: an array of y-coordinate(s) of detected objects in pixels

    Return:
        a 2d array of shape [len(x1), len(x2)] of L2 pixel distances between targets and detected objects

    The funciton works even if the number of target objects is different from number of output
    objects. Note that i-th row and j-th column of the output matrix denotes the distance
    between the i-th target object and j-th predicted object.
    """
    assert (
        x1.shape == y1.shape and x2.shape == y2.shape
    ), "Shapes of corresponding arrays must be the same."
    target_vectors = np.stack((x1, y1), axis=1)
    prediction_vectors = np.stack((x2, y2), axis=1)
    return spatial.distance_matrix(target_vectors, prediction_vectors)


class Matching(ABC):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize matching class."""

    @abstractmethod
    def preprocess_catalog(self, catalog: Table) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts coordinate information required for matching."""

    def compute_distance_matrix(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        """Based on the catalogs and user-defined preprocessing computes distance matrix."""
        x1, y1 = self.preprocess_catalog(truth_catalog)
        x2, y2 = self.preprocess_catalog(predicted_catalog)
        return self.distance_matrix_function(x1, y1, x2, y2)

    @abstractmethod
    def __call__(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        # TODO: explain the output of the match method here or in child classes?
        """Perform matching procedure between truth and prediction"""


class IdentityMatching(Matching):
    """Assuming that catalogs are already matched one-to-one, performs trivial identity matching;"""

    def __call__(self, truth_catalog, predicted_catalog) -> np.ndarray:
        return np.array(range(len(predicted_catalog)))


class PixelHungarianMatching(Matching):
    def __init__(
        self, pixel_max_sep=5.0, *args, **kwargs
    ) -> None:
        """Initialize matching class.

        Args:
            max_sep: the maximum separation in pixels to be considered a match
        """
        self.distance_function = pixel_l2_distance_matrix
        self.max_sep = pixel_max_sep

    def preprocess_catalog(self, catalog: Table) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pixel coordinates out of catalogs"""
        if "x_peak" not in catalog.colnames:
            raise KeyError("One of the catalogs has no column 'x_peak'")
        if "y_peak" not in catalog.colnames:
            raise KeyError("One of the catalogs has no column 'y_peak'")
        return (catalog["x_peak"], catalog["y_peak"])

    def __call__(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        """Performs Hungarian matching algorithm on pixel coordinates from catalogs.

        Based on this implementation in scipy:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

        Args:
            truth_catalog: truth catalog containing relevant detecion information
            predicted_catalog: predicted catalog to compare with the ground truth
        Returns:
            matches: a 1D array where j-th entry is the index of the target row 
                that matched with the j-th detected row. If no match, value is -1.
        """
        dist = self.compute_distance_matrix(truth_catalog, predicted_catalog)
        # solve optimization problem using Hungarian matching algorithm
        # truth_catalog[true_indx[i]] is matched with predicted_catalog[detected_indx[i]]
        # len(true_indx) = len(detect_indx) = min(len(true_table), len(detected_table))
        true_indx, detected_indx = linear_sum_assignment(dist)

        # if the distance is greater than max_sep then mark detection as -1
        mask = dist[true_indx, detected_indx] > self.max_sep
        detected_indx[mask] = -1
        return detected_indx

class ClosestSkyNeighbourMatching(Matching):
    def __init__(self, arcsec_max_sep=2.0, *args, **kwargs) -> None:
        """Initialize matching class.

        Args:
            max_sep: the maximum separation in arcsec to be considered a match
            num_neighbours: number of closest neighbours to consider for each target
        """
        self.max_sep = arcsec_max_sep

    def preprocess_catalog(self, catalog: Table) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ra, dec coordinates out of catalogs"""
        if "ra" not in catalog.colnames:
            raise KeyError("One of the catalogs has no column 'ra'")
        if "dec" not in catalog.colnames:
            raise KeyError("One of the catalogs has no column 'dec'")
        return (catalog["ra"], catalog["dec"])
    
    def __call__(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        """
        Performs 1st Nearest Neigbour look up for each coordinate in predicted_catalog.
        Then we prune repeated detections of a given target source by assigning it
        to the closest object in the predicted catalog, and discarding the rest. Finally,
        we apply max_separation threshold.

        Based on this implementation in AstroPy:
        https://docs.astropy.org/en/stable/coordinates/matchsep.html

        Args:
            truth_catalog: truth catalog containing relevant detecion information
            predicted_catalog: predicted catalog to compare with the ground truth
        Returns:
            matches: a 1D array where j-th entry is the index of the target row 
                that matched with the j-th detected row. If no match, value is -1.
        """
        ra1, dec1 = self.preprocess_catalog(truth_catalog)
        ra2, dec2 = self.preprocess_catalog(predicted_catalog)
        true_coordinates = SkyCoord(ra = ra1*units.arcsec, dec=dec1*units.arcsec)
        pred_coordinates = SkyCoord(ra = ra2*units.arcsec, dec=dec2*units.arcsec)
        
        # computes 1st nearest neighbour
        idx, d2d, _ = pred_coordinates.match_to_catalog_sky(true_coordinates)

        # remove repeated detecions, saving only closest one
        detected_indx = np.array([-1] * len(idx))
        for target_idx in set(idx):
            masked_d2d = d2d.arcsec.copy()
            masked_d2d[idx != target_idx] = np.inf
            match_id = np.argmin(masked_d2d)
            detected_indx[match_id] = target_idx

        # if the matched distance exceeds max_sep, we discard that detection
        detected_indx[d2d.to(units.arcsec) > self.max_sep] = -1
        return detected_indx