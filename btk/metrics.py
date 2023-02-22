"""Implements a variety of metrics for evaluating measurement results in BTK."""
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from btk.draw_blends import BlendBatch
from btk.measure import MeasuredBatch
from btk.metrics_utils import get_matches, match_stats


class Metric(ABC):
    """Abstract class for BTK metrics."""

    def __init__(self, batch_size: int, survey_name: str, *args) -> None:
        """Initialize metric."""
        self.batch_size = batch_size
        self.all_data = []
        self.values = []
        self.survey_name = survey_name

    @abstractmethod
    def _compute(self, data: tuple) -> Union[np.ndarray, float]:
        """Compute metric based on data."""

    @abstractmethod
    def _get_data(
        self, blend_batch: BlendBatch, measured_batch: MeasuredBatch, *args, matches=None, **kwargs
    ) -> tuple:
        """Get data necessary to compute metric."""
        assert self.survey_name in blend_batch.results
        assert self.survey_name == measured_batch.survey_name

    @abstractmethod
    def aggregate(self) -> Union[np.ndarray, float]:
        """Aggregate self.values to compute metric over multiple batches."""

    def reset(self) -> None:
        """Reset global state."""
        self.values = []
        self.all_data = []

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Compute metric on batch, update global state, and return metric."""
        data = self._get_data(*args, **kwargs)
        value = self._compute(data)
        self.all_data.append(data)
        self.values.append(value)
        return value


class Precision(Metric):
    """Precision class metric."""

    def _get_data(
        self, blend_batch: BlendBatch, measured_batch: MeasuredBatch, matches=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute precision on batch."""
        super().get_data(blend_batch, measured_batch, matches=matches)

        if matches is None:
            matches = []
            for ii in range(self.batch_size):
                truth = blend_batch.results["catalog"][self.survey_name][ii]
                pred = measured_batch.catalog[ii]
                matches[ii] = get_matches(truth, pred)

        return match_stats(matches)  # tp, fp, t, p

    def _compute(self, data: tuple) -> float:
        tp, _, _, p = data
        return tp / p

    def aggregate(self) -> float:
        """Average all precisions so far."""
        return np.mean(self.values)
