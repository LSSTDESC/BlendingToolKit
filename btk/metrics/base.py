"""Implements base class for metrics in BTK."""
# pylint: disable=arguments-differ
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np


class Metric(ABC):
    """Abstract class for BTK metrics."""

    def __init__(self, batch_size: int, *args, **kwargs) -> None:
        """Initialize metric."""
        self.batch_size = batch_size
        self.all_data = []
        self.values = []

    @abstractmethod
    def _compute(self, data: dict) -> Union[np.ndarray, float]:
        """Compute metric based on data."""

    @abstractmethod
    def _get_data(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Get data necessary to compute metric."""

    def aggregate(self) -> Union[np.ndarray, float]:
        """Aggregate `self.values` to compute metric over multiple batches."""
        return np.mean(self.values)

    def reset(self) -> None:
        """Reset global state."""
        self.values = []
        self.all_data = []

    def __call__(self, *args, **kwargs) -> Union[np.ndarray, float]:
        """Compute metric on batch, update global state, and return metric.

        Assumes that `deblend_batch` has been matched to `blend_batch` with the corresponding
        matching information in `match_list`.
        """
        data = self._get_data(*args, **kwargs)
        value = self._compute(data)
        self.all_data.append(data)
        self.values.append(value)
        return value
