"""Metrics for reconstruction."""
from abc import ABC
from typing import Callable, Dict

import numpy as np

from btk.metrics.base import Metric
from btk.metrics.utils import mse, psnr, struct_sim


class ReconstructionMetric(Metric, ABC):
    """Abstract class for reconstruction metrics in BTK."""

    def _get_recon_metric(
        self, iso_images1: np.ndarray, iso_images2: np.ndarray, metric_func: Callable
    ) -> np.ndarray:
        assert iso_images1.shape == iso_images2.shape
        assert iso_images1.ndim == 4  # batch, max_n_sources, x, y
        results = []
        for ii in range(self.batch_size):
            n_sources1 = np.sum(iso_images1[ii].sum(axis=(-1, -2)) > 0).astype(int)
            n_sources2 = np.sum(iso_images2[ii].sum(axis=(-1, -2)) > 0).astype(int)
            n_sources = min(n_sources1, n_sources2)  # just in case
            if n_sources > 0:
                images1 = iso_images1[ii, :n_sources]
                images2 = iso_images2[ii, :n_sources]
                mets = metric_func(images1, images2)
                assert mets.shape == (n_sources,)
                for ii in range(n_sources):
                    results.append(mets[ii])

        return np.array(results)

    def _get_data(self, iso_images1: np.ndarray, iso_images2: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute reconstruction metric based on isolated or deblended images."""


class MSE(ReconstructionMetric):
    """MSE class metric."""

    def _get_data(self, iso_images1: np.ndarray, iso_images2: np.ndarray) -> Dict[str, np.ndarray]:
        return {"mse": self._get_recon_metric(iso_images1, iso_images2, mse)}

    def _compute(self, data: dict) -> float:
        return np.mean(data["mse"])


class PSNR(ReconstructionMetric):
    """PSNR class metric."""

    def _get_data(self, iso_images1: np.ndarray, iso_images2: np.ndarray) -> Dict[str, np.ndarray]:
        return {"psnr": self._get_recon_metric(iso_images1, iso_images2, psnr)}

    def _compute(self, data: dict) -> float:
        return np.mean(data["psnr"])


class StructSim(ReconstructionMetric):
    """Structurality Similarity class metric."""

    def _get_data(self, iso_images1: np.ndarray, iso_images2: np.ndarray) -> Dict[str, np.ndarray]:
        return {"ssim": self._get_recon_metric(iso_images1, iso_images2, struct_sim)}

    def _compute(self, data: dict) -> float:
        return np.mean(data["ssim"])
