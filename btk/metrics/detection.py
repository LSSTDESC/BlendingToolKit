"""Implementation of detection metrics."""
from typing import Dict

import numpy as np

from btk.metrics.base import Metric
from btk.metrics.utils import effmat


class DetectionMetric(Metric):
    """Base class for detection metrics."""

    def _get_data(
        self, n_matches: np.ndarray, n_true: np.ndarray, n_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute detection metric on batch."""
        raise NotImplementedError


class Precision(DetectionMetric):
    """Precision class metric."""

    def _get_data(
        self, n_matches: np.ndarray, n_true: np.ndarray, n_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute precision on batch."""
        tp = n_matches
        fp = n_pred - n_matches
        t = n_true
        p = n_pred
        return {"tp": tp, "fp": fp, "t": t, "p": p}

    def _compute(self, data: tuple) -> float:
        return data["tp"].sum() / data["p"].sum()


class Recall(Precision):
    """Recall class metric."""

    def _compute(self, data: tuple) -> float:
        return data["tp"].sum() / data["t"].sum()


class F1(Precision):
    """F1 class metric."""

    def _compute(self, data: tuple) -> float:
        precision = data["tp"].sum() / data["p"].sum()
        recall = data["tp"].sum() / data["t"].sum()
        return 2 * (precision**-1 + recall**-1) ** -1


class Efficiency(DetectionMetric):
    """Efficiency class metric."""

    def _get_data(
        self, n_matches: np.ndarray, n_true: np.ndarray, n_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute efficiency matrix on batch."""
        tp = n_matches
        t = n_true
        eff_matrix = effmat(tp, t)
        return {"eff_matrix": eff_matrix}

    def _compute(self, data: dict) -> np.ndarray:
        return data["eff_matrix"]

    def aggregate(self) -> np.ndarray:
        """Aggregate efficiency matrices over multiple batches."""
        max_r = max(eff.shape[0] for eff in self.values)
        max_c = max(eff.shape[1] for eff in self.values)
        final = np.zeros((max_r, max_c))
        for eff in self.values:
            final[: eff.shape[0], : eff.shape[1]] += eff
        return final
