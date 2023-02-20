"""Implements a variety of metrics for evaluating measurement results in BTK."""
from abc import ABC, abstractmethod

import numpy as np

from btk.draw_blends import BlendBatch
from btk.measure import MeasuredBatch
from btk.metrics_utils import get_matches


class Metric(ABC):
    """Abstract class for BTK metrics."""

    def __init__(self, batch_size: int, survey_name: str, *args) -> None:
        """Initialize metric."""
        self.batch_size = batch_size
        self.values = []
        self.survey_name = survey_name

    @abstractmethod
    def compute(
        self, blend_batch: BlendBatch, measured_batch: MeasuredBatch, *args, matches=None, **kwargs
    ) -> np.ndarray:
        """Compute metric based on blend and measured batches."""
        assert self.survey_name in blend_batch.results
        assert self.survey_name == measured_batch.survey_name

    @abstractmethod
    def aggregate(self) -> np.ndarray:
        """Aggregate self.values to compute metric over multiple batches."""
        pass

    def reset(self) -> None:
        """Reset global state."""
        self.values = []

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Compute metric on batch, update global state, and return metric on batch."""
        value = self.compute(*args, **kwargs)
        self.values.append(value)
        return value


class Precision(Metric):
    """Precision class metric."""

    def compute(
        self, blend_batch: BlendBatch, measured_batch: MeasuredBatch, matches=None
    ) -> np.ndarray:
        """Compute precision on batch."""
        if matches is None:
            matches = []
            for ii in range(self.batch_size):
                truth = blend_batch.results["catalog"][self.survey_name][ii]
                pred = measured_batch.catalog[ii]
                matches[ii] = get_matches(truth, pred)
