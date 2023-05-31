"""Implements a variety of metrics for evaluating measurement results in BTK."""
# pylint: disable=arguments-differ
from abc import ABC, abstractmethod
from typing import Callable, Dict, Union

import numpy as np

from btk.deblend import DeblendedBatch
from btk.draw_blends import BlendBatch
from btk.match import IdentityMatching
from btk.metrics_utils import iou, match_stats


class Metric(ABC):
    """Abstract class for BTK metrics."""

    def __init__(self, batch_size: int, survey_name: str, *args, **kwargs) -> None:
        """Initialize metric."""
        self.batch_size = batch_size
        self.all_data = []
        self.values = []
        self.survey_name = survey_name

    @abstractmethod
    def _compute(self, data: dict) -> Union[np.ndarray, float]:
        """Compute metric based on data."""

    @abstractmethod
    def _get_data(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Get data necessary to compute metric."""

    @abstractmethod
    def aggregate(self) -> Union[np.ndarray, float]:
        """Aggregate self.values to compute metric over multiple batches."""

    def reset(self) -> None:
        """Reset global state."""
        self.values = []
        self.all_data = []

    def __call__(
        self,
        blend_batch: BlendBatch,
        deblended_batch: DeblendedBatch,
        *args,
        match_fn=IdentityMatching,
        **kwargs,
    ) -> np.ndarray:
        """Compute metric on batch, update global state, and return metric."""
        assert self.survey_name in blend_batch.results
        assert self.survey_name == deblended_batch.survey_name
        data = self._get_data(blend_batch, deblended_batch, *args, match_fn=match_fn, **kwargs)
        value = self._compute(data)
        self.all_data.append(data)
        self.values.append(value)
        return value


class Precision(Metric):
    """Precision class metric."""

    def _get_data(
        self,
        blend_batch: BlendBatch,
        measured_batch: DeblendedBatch,
        match_fn: Callable = IdentityMatching,
    ) -> Dict[str, np.ndarray]:
        """Compute precision on batch."""
        matches = []
        n_preds = np.zeros(self.batch_size)
        for ii in range(self.batch_size):
            truth = blend_batch.results["catalog"][self.survey_name][ii]
            pred = measured_batch.catalog_list[ii]
            n_preds[ii] = len(pred)
            matches[ii], _ = match_fn(truth, pred)

        tp, fp, t, p = match_stats(matches, n_preds)
        return {"tp": tp, "fp": fp, "t": t, "p": p}

    def _compute(self, data: tuple) -> float:
        return data["tp"].sum() / data["p"].sum()

    def aggregate(self) -> float:
        """Average all precisions so far."""
        return np.mean(self.values)


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


class IoU(Metric):
    """Intersection-over-Union class metric."""

    # TODO: Need to prompt user to add segmentations to `blend_batch`.

    def _get_data(
        self,
        blend_batch: BlendBatch,
        measured_batch: DeblendedBatch,
        match_fn=IdentityMatching,
    ) -> Dict[str, np.ndarray]:
        assert "segmentation" in blend_batch.results[self.survey_name]

        matched_segs = {"seg1": [], "seg2": []}
        for ii in range(self.batch_size):
            truth = blend_batch.results["catalog"][self.survey_name][ii]
            pred = measured_batch.catalog_list[ii]
            match, _ = match_fn(truth, pred)
            for jj in range(len(match)):
                seg1 = blend_batch.results[self.survey_name]["segmentation"][ii][jj]
                seg2 = measured_batch.segmentation[ii][match[jj]]
                matched_segs["seg1"].append(seg1)
                matched_segs["seg2"].append(seg2)

        matched_segs["seg1"] = np.array(matched_segs["seg1"])
        matched_segs["seg2"] = np.array(matched_segs["seg2"])

        n, h, w = matched_segs["seg1"]
        assert (n, h, w) == matched_segs["seg2"].shape

        return matched_segs

    def _compute(self, data: dict) -> float:
        return iou(data["seg1"], data["seg2"]).mean()
