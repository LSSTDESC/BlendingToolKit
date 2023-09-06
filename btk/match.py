"""Tools to match detected objects with truth catalog."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy import spatial
from scipy.optimize import linear_sum_assignment


class Matching:
    """Stores information about matching between truth and detected objects for a single batch."""

    def __init__(
        self,
        true_matches: List[np.ndarray],
        pred_matches: List[np.ndarray],
        n_true: np.ndarray,
        n_pred: np.ndarray,
    ) -> None:
        """Initialize MatchInfo.

        Args:
            true_matches: a list of 1D array, each entry corresponds to a numpy array
                containing the index of detected object in the truth catalog that
                got matched with the i-th truth object in the blend.
            pred_matches: a list of 1D array, where the j-th entry of i-th array
                corresponds to the index of truth object in the i-th blend
                that got matched with the j-th detected object in that blend.
                If no match, value is -1.
            n_true: a 1D array of length N, where each entry is the number of truth objects.
            n_pred: a 1D array of length N, where each entry is the number of detected objects.
        """
        self.true_matches = true_matches
        self.pred_matches = pred_matches
        self.n_true = n_true
        self.n_pred = n_pred
        self.batch_size = len(n_true)
        self.max_n_sources = max(len(d) for d in self.true_matches)

    def _match_catalogs(self, catalog_list: Table, true_or_pred: str) -> List[Table]:
        if true_or_pred == "true":
            matches = self.true_matches
        elif true_or_pred == "pred":
            matches = self.pred_matches
        else:
            raise ValueError("true_or_pred must be 'true' or 'pred'")

        matched_catalogs = []
        for ii in range(self.batch_size):
            cat = catalog_list[ii].copy()
            matched_catalogs.append(cat[matches[ii]])
        return matched_catalogs

    def _match_arrays(self, *arrs: np.ndarray, true_or_pred: str) -> tuple:
        if true_or_pred == "true":
            matches = self.true_matches
        elif true_or_pred == "pred":
            matches = self.pred_matches
        else:
            raise ValueError("true_or_pred must be 'true' or 'pred'")

        new_arrs = []
        for arr in arrs:
            assert len(arr) == self.batch_size
            new_arr = np.zeros_like(arr)
            for ii in range(self.batch_size):
                n_sources = len(matches[ii])
                assert n_sources <= self.max_n_sources
                new_arr[ii, :n_sources] = arr[ii][matches[ii]]
            new_arrs.append(new_arr[:, : self.max_n_sources])
        return tuple(new_arrs) if len(new_arrs) > 1 else new_arrs[0]

    def match_true_catalogs(self, catalog_list: Table) -> List[Table]:
        """Returns a list of matched truth catalogs."""
        return self._match_catalogs(catalog_list, true_or_pred="true")

    def match_true_arrays(self, *arrs: np.ndarray) -> tuple:
        """Return matched truth arrays."""
        return self._match_arrays(*arrs, true_or_pred="true")

    def match_pred_catalogs(self, catalog_list: Table) -> List[Table]:
        """Returns a list of matched predicted catalogs."""
        return self._match_catalogs(catalog_list, true_or_pred="pred")

    def match_pred_arrays(self, *arrs: np.ndarray) -> tuple:
        """Return matched predicted arrays."""
        return self._match_arrays(*arrs, true_or_pred="pred")

    def filter_by_true(self, mask: List[np.ndarray]) -> "Matching":
        """Returns a new Matching object with true objects that pass the mask."""
        new_true_matches = []
        new_pred_matches = []
        new_n_true = []
        new_n_pred = []
        for ii in range(self.batch_size):
            true_match = self.true_matches[ii]
            pred_match = self.pred_matches[ii]

            # get indices in true_matches of true objects that do not pass mask
            isin_true = np.isin(true_match, np.where(~mask[ii])[0])
            index_of_interest = np.argwhere(isin_true).ravel()

            # remove those indices from true_matches and pred_matches
            new_true_matches.append(np.delete(true_match, index_of_interest, axis=0))
            new_pred_matches.append(np.delete(pred_match, index_of_interest, axis=0))

            new_n_true.append(np.sum(mask[ii]))
            new_n_pred.append(self.n_pred[ii])

        return Matching(
            new_true_matches, new_pred_matches, np.array(new_n_true), np.array(new_n_pred)
        )

    @property
    def tp(self) -> np.ndarray:
        """Returns true positive array."""
        return np.array([len(d) for d in self.true_matches])

    @property
    def fp(self) -> np.ndarray:
        """Returns false positive array."""
        return self.n_pred - self.tp

    @property
    def t(self) -> np.ndarray:
        """Returns truth array."""
        return self.n_true

    @property
    def p(self) -> np.ndarray:
        """Returns positive array."""
        return self.n_pred


class Matcher(ABC):
    """Base class for matching algorithms."""

    def __init__(self, **kwargs) -> None:  # pylint: disable=unused-argument
        """Initialize matching class."""
        self.distance_matrix_function = pixel_l2_distance_matrix

    def __call__(self, true_catalog_list: List[Table], pred_catalog_list: List[Table]) -> Matching:
        """Performs matching procedure between truth and prediction catalog lists."""
        match_true = []
        match_pred = []
        n_true = []
        n_pred = []
        for true_catalog, pred_catalog in zip(true_catalog_list, pred_catalog_list):
            if len(true_catalog) == 0 or len(pred_catalog) == 0:
                true_match, pred_match = np.array([]).astype(int), np.array([]).astype(int)
            else:
                true_match, pred_match = self.match_catalogs(true_catalog, pred_catalog)
            match_true.append(true_match)
            match_pred.append(pred_match)
            n_true.append(len(true_catalog))
            n_pred.append(len(pred_catalog))
        return Matching(match_true, match_pred, np.array(n_true), np.array(n_pred))

    def compute_distance_matrix(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        """Based on the catalogs and user-defined preprocessing computes distance matrix."""
        x1, y1 = self.preprocess_catalog(truth_catalog)
        x2, y2 = self.preprocess_catalog(predicted_catalog)
        return self.distance_matrix_function(x1, y1, x2, y2)

    def preprocess_catalog(self, catalog: Table) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts coordinate information required for matching."""
        if "x_peak" not in catalog.colnames or "y_peak" not in catalog.colnames:
            raise KeyError("One of the catalogs has no column 'x_peak' or 'y_peak'")
        return (catalog["x_peak"], catalog["y_peak"])

    @abstractmethod
    def match_catalogs(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        """Perform matching procedure between truth and prediction."""


class IdentityMatcher(Matcher):
    """Assumes catalogs are already matched one-to-one, returns trivial identity matching."""

    def match_catalogs(self, truth_catalog, predicted_catalog) -> np.ndarray:
        """Returns trivial identity matching."""
        true_indx = np.array(range(len(truth_catalog)))
        pred_indx = np.array(range(len(predicted_catalog)))
        return true_indx, pred_indx


class PixelHungarianMatcher(Matcher):
    """Match based on pixel coordinates using Hungarian matching algorithm."""

    def __init__(self, pixel_max_sep=5.0, **kwargs) -> None:
        """Initialize matching class.

        Args:
            pixel_max_sep: the maximum separation in pixels to be considered a match
        """
        super().__init__(**kwargs)
        self.distance_function = pixel_l2_distance_matrix
        self.max_sep = pixel_max_sep

    def match_catalogs(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        """Performs Hungarian matching algorithm on pixel coordinates from catalogs.

        Based on this implementation in scipy:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

        Args:
            truth_catalog: truth catalog containing relevant detecion information
            predicted_catalog: predicted catalog to compare with the ground truth
        Returns:
            matched_indx: a 1D array where j-th entry is the index of the target row
                that matched with the j-th detected row. If no match, value is -1.
        """
        dist = self.compute_distance_matrix(truth_catalog, predicted_catalog)
        # solve optimization problem using Hungarian matching algorithm
        # truth_catalog[true_indx[i]] is matched with predicted_catalog[matched_indx[i]]
        # len(true_indx) = len(detect_indx) = min(len(true_table), len(detected_table))
        true_indx, pred_indx = linear_sum_assignment(dist)

        # if the distance is greater than max_sep then mark detection as -1
        true_mask = dist.T[pred_indx, true_indx] > self.max_sep
        true_indx[true_mask] = -1
        pred_mask = dist[true_indx, pred_indx] > self.max_sep
        pred_indx[pred_mask] = -1

        return true_indx, pred_indx


class SkyClosestNeighbourMatcher(Matcher):
    """Match based on closest neighbour in the sky."""

    def __init__(self, arcsec_max_sep=2.0, **kwargs) -> None:
        """Initialize matching class.

        Args:
            arcsec_max_sep: the maximum separation in arcsec to be considered a match
        """
        super().__init__(**kwargs)
        self.max_sep = arcsec_max_sep

    def preprocess_catalog(self, catalog: Table) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ra, dec coordinates out of catalogs."""
        if "ra" not in catalog.colnames or "dec" not in catalog.colnames:
            raise KeyError("One of the catalogs has no column 'ra' or 'dec'")
        return (catalog["ra"], catalog["dec"])

    def match_catalogs(self, truth_catalog: Table, predicted_catalog: Table) -> np.ndarray:
        """Matches catalogs based on closest neighbour in sky coordinates.

        Performs 1st Nearest Neigbour look up for each coordinate in predicted_catalog.
        Then we prune repeated detections of a given target source by assigning it
        to the closest object in the predicted catalog, and discarding the rest. Finally,
        we apply `max_sep` threshold.

        Based on this implementation in AstroPy:
        https://docs.astropy.org/en/stable/coordinates/matchsep.html

        Args:
            truth_catalog: truth catalog containing relevant detecion information
            predicted_catalog: predicted catalog to compare with the ground truth

        Returns:
            matched_indx: a 1D array where j-th entry is the index of the target row
                that matched with the j-th detected row. If no match, value is -1.
        """
        ra1, dec1 = self.preprocess_catalog(truth_catalog)
        ra2, dec2 = self.preprocess_catalog(predicted_catalog)
        true_coordinates = SkyCoord(ra=ra1 * units.arcsec, dec=dec1 * units.arcsec)
        pred_coordinates = SkyCoord(ra=ra2 * units.arcsec, dec=dec2 * units.arcsec)

        # computes 1st nearest neighbour
        idx, d2d, _ = pred_coordinates.match_to_catalog_sky(true_coordinates)

        # remove repeated detecions, saving only closest one
        pred_indx = np.array([-1] * len(idx))
        for target_idx in set(idx):
            masked_d2d = d2d.arcsec.copy()
            masked_d2d[idx != target_idx] = np.inf
            match_id = np.argmin(masked_d2d)
            pred_indx[match_id] = target_idx

        # if the matched distance exceeds max_sep, we discard that detection
        pred_indx[d2d.to(units.arcsec) > self.max_sep * units.arcsec] = -1

        # now for ture indices
        idx, d2d, _ = true_coordinates.match_to_catalog_sky(pred_coordinates)
        true_indx = np.array([-1] * len(idx))
        for target_idx in set(idx):
            masked_d2d = d2d.arcsec.copy()
            masked_d2d[idx != target_idx] = np.inf
            match_id = np.argmin(masked_d2d)
            true_indx[match_id] = target_idx

        true_indx[d2d.to(units.arcsec) > self.max_sep * units.arcsec] = -1

        return true_indx, pred_indx


def pixel_l2_distance_matrix(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> np.ndarray:
    """Computes the pixel L2 (Ecludian) distance matrix between targets and detections.

    Args:
        x1: an array of x-coordinate(s) of target objects in pixels
        y1: an array of y-coordinate(s) of target objects in pixels
        x2: an array of x-coordinate(s) of detected objects in pixels
        y2: an array of y-coordinate(s) of detected objects in pixels

    Return:
        2d array of shape [len(x1), len(x2)] of L2 pixel distances between
        targets and detected objects.

    The funciton works even if the number of target objects is different from number of output
    objects. Note that i-th row and j-th column of the output matrix denotes the distance
    between the i-th target object and j-th predicted object.
    """
    assert x1.shape == y1.shape and x2.shape == y2.shape, "Shapes of arrays must be the same."
    target_vectors = np.stack((x1, y1), axis=1)
    prediction_vectors = np.stack((x2, y2), axis=1)
    return spatial.distance_matrix(target_vectors, prediction_vectors)
