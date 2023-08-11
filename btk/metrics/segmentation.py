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
    """Intersection-over-Union class metric."""

    def _get_data(
        self,
        seg1: np.ndarray,
        seg2: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        assert seg1.shape == seg2.shape
        assert seg1.ndim == 4  # batch, max_n_sources, x, y
        ious = np.zeros(self.batch_size)
        for ii in range(self.batch_size):
            n_sources = np.sum(~np.isnan(seg1[ii].sum(axis=(-1, -2))))
            if n_sources > 0:
                seg1_ii = seg1[ii, :n_sources]
                seg2_ii = seg2[ii, :n_sources]
                ious[ii] = iou(seg1_ii, seg2_ii)
            else:
                ious[ii] = np.nan

        return {"iou": ious}

    def _compute(self, data: dict) -> float:
        return np.nanmean(data["iou"])
