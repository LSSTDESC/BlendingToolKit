"""Implements segmentation metrics for evaluating deblending results in BTK."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from btk.metrics.base import Metric
from btk.metrics.utils import iou


class SegmentationMetric(Metric, ABC):
    """Abstract class for segmentation metrics in BTK."""

    @abstractmethod
    def _get_data(
        self,
        seg1: np.ndarray,
        seg2: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute segmentation metric on batch."""


class IoU(SegmentationMetric):
    """Intersection-over-Union class metric.

    Note that this metric assumes that the input segmentation arrays are booleans. An error is
    raised if that condition is not met.
    """

    def _get_data(
        self,
        seg1: np.ndarray,
        seg2: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        assert seg1.shape == seg2.shape
        assert seg1.ndim == 4  # batch, max_n_sources, x, y

        if not (seg1.dtype == "bool" and seg2.dtype == "bool"):
            raise TypeError("Segmentation arrays for the IoU metric should be of boolean type.")

        ious = np.full((self.batch_size, seg1.shape[1]), fill_value=np.nan)
        for ii in range(self.batch_size):
            n_sources1 = np.sum(np.sum(seg1[ii], axis=(-1, -2)) > 0)
            n_sources2 = np.sum(np.sum(seg2[ii], axis=(-1, -2)) > 0)
            n_sources = min(n_sources1, n_sources2)
            if n_sources > 0:
                seg1_ii = seg1[ii, :n_sources]
                seg2_ii = seg2[ii, :n_sources]
                ious[ii, :n_sources] = iou(seg1_ii, seg2_ii)

        return {"iou": ious}

    def _compute(self, data: dict) -> float:
        return np.nanmean(data["iou"])
