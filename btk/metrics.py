"""Implements a variety of metrics for evaluating measurement results in BTK."""
# pylint: disable=arguments-differ
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

from btk.blend_batch import BlendBatch, DeblendBatch
from btk.metrics_utils import iou, match_stats, mse, psnr, struct_sim


class Metric(ABC):
    """Abstract class for BTK metrics."""

    def __init__(self, batch_size: int, band_idx: int, *args, **kwargs) -> None:
        """Initialize metric."""
        self.batch_size = batch_size
        self.band_idx = band_idx
        self.all_data = []
        self.values = []

    @abstractmethod
    def _compute(self, data: dict) -> Union[np.ndarray, float]:
        """Compute metric based on data."""

    @abstractmethod
    def _get_data(
        self, blend_batch: BlendBatch, deblend_batch: DeblendBatch, match_list: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Get data necessary to compute metric."""

    @abstractmethod
    def aggregate(self) -> Union[np.ndarray, float]:
        """Aggregate self.values to compute metric over multiple batches."""

    def reset(self) -> None:
        """Reset global state."""
        self.values = []
        self.all_data = []

    def __call__(
        self, blend_batch: BlendBatch, deblend_batch: DeblendBatch, match_list: List[np.ndarray]
    ) -> np.ndarray:
        """Compute metric on batch, update global state, and return metric.

        Assumes that `deblend_batch` has been matched to `blend_batch` with the corresponding
        matching information in `match_list`.
        """
        data = self._get_data(blend_batch, deblend_batch, match_list)
        value = self._compute(data)
        self.all_data.append(data)
        self.values.append(value)
        return value


class Precision(Metric):
    """Precision class metric."""

    def _get_data(
        self,
        blend_batch: BlendBatch,
        deblend_batch: DeblendBatch,
        match_list: List[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute precision on batch."""
        n_preds = np.zeros(self.batch_size)
        for ii in range(self.batch_size):
            pred = len(deblend_batch.catalog_list[ii])
            n_preds[ii] = len(pred)

        tp, fp, t, p = match_stats(match_list, n_preds)
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

    def _get_data(
        self,
        blend_batch: BlendBatch,
        deblend_batch: DeblendBatch,
        match_list: List[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        assert "segmentation" in blend_batch and blend_batch.segmentation is not None
        assert "segmentation" in deblend_batch and deblend_batch.segmentation is not None

        ious = np.zeros(self.batch_size)
        for ii in range(self.batch_size):
            n_match = sum(match_list[ii] > 0)
            if n_match > 0:  # TODO: is this the correct way to handle empty matches?
                seg1 = blend_batch.segmentation[ii, :n_match]
                seg2 = deblend_batch.segmentation[ii, :n_match]
                ious[ii] = iou(seg1, seg2)
            else:
                ious[ii] = np.nan  # skip over batch with no matches in computing iou

        return {"iou": ious}

    def _compute(self, data: dict) -> float:
        return np.nanmean(data["iou"])


class MSE(Metric):
    """Intersection-over-Union class metric."""

    def _get_data(
        self, blend_batch: BlendBatch, deblend_batch: DeblendBatch, match_list: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        assert "deblended_images" in deblend_batch
        assert deblend_batch.deblended_images is not None

        mses = np.zeros(self.batch_size)
        for ii in range(self.batch_size):
            n_match = sum(match_list[ii] > 0)
            if n_match > 0:
                images1 = blend_batch.isolated_images[ii, :n_match, self.band_idx]
                images2 = deblend_batch.deblended_images[ii, :n_match]
                mses[ii] = mse(images1, images2)
            else:
                mses[ii] = np.nan

        return {"mse": mses}

    def _compute(self, data: dict) -> float:
        return np.nanmean(data["mse"])


class PSNR(Metric):
    """Intersection-over-Union class metric."""

    def _get_data(
        self, blend_batch: BlendBatch, deblend_batch: DeblendBatch, match_list: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        assert "deblended_images" in deblend_batch
        assert deblend_batch.deblended_images is not None

        psnrs = np.zeros(self.batch_size)
        for ii in range(self.batch_size):
            n_match = sum(match_list[ii] > 0)
            if n_match > 0:
                images1 = blend_batch.isolated_images[ii, :n_match, self.band_idx]
                images2 = deblend_batch.deblended_images[ii, :n_match]
                psnrs[ii] = psnr(images1, images2)
            else:
                psnrs[ii] = np.nan

        return {"psnr": psnrs}

    def _compute(self, data: dict) -> float:
        return np.nanmean(data["psnr"])


class StructSim(Metric):
    """Intersection-over-Union class metric."""

    def _get_data(
        self, blend_batch: BlendBatch, deblend_batch: DeblendBatch, match_list: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        assert "deblended_images" in deblend_batch
        assert deblend_batch.deblended_images is not None

        ssims = np.zeros(self.batch_size)
        for ii in range(self.batch_size):
            n_match = sum(match_list[ii] > 0)
            if n_match > 0:
                images1 = blend_batch.isolated_images[ii, :n_match, self.band_idx]
                images2 = deblend_batch.deblended_images[ii, :n_match]
                ssims[ii] = struct_sim(images1, images2)
            else:
                ssims[ii] = np.nan

        return {"ssim": ssims}

    def _compute(self, data: dict) -> float:
        return np.nanmean(data["ssim"])
