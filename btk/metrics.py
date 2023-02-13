r"""Implements a variety of metrics for evaluating measurement results in BTK.

BTK users are expected to use the MetricsGenerator class, which is initialized by providing
a MeasureGenerator as well as some parameters. Users which do not want to use the full BTK
pipeline may use the compute_metrics function which takes the raw data as an input.

Currently, we support the following metrics:

* For detection, all metrics are per batch:

  * Number of true positives, ie number of true galaxies which have been correctly detected
  * Number of false positives, ie number of detected galaxies which do not correspond
    to a true galaxy
  * Number of false negatives, ie number of true galaxies which have not been detected
  * Precision, the ratio of true positives against the total number of positives; describes
    how much the algorithm is susceptible to make false detections (closer to 1 is better)
  * Recall, the ratio of true positives against the number of true galaxies (which is equal
    to true positives + false negatives); indicates the capacity of the algorithm for
    detecting all the galaxies (closer to 1 is better)
  * F1 score, the harmonic mean of precision and recall; gives an overall assessment of the
    detection (closer to 1 is better)
  * Efficiency matrix, contains for each possible number of true galaxies in a blend the
    distribution of the number of detected galaxies in blends containing this number of true
    galaxies.

* For segmentation, all metrics are per galaxy:

  * Intersection-over-Union (IoU), the ratio between the intersection of the true and
    detected segmentations (true segmentation is computed by applying a threshold on
    the true image) and the union of the two. Closer to 1 is better.

* For reconstruction, all metrics are per galaxy:

  * Mean Square Residual (MSR), the mean square error between the true image and the
    deblended image. Lower is better.
  * Peak Signal to Noise Ratio (PSNR), the log of the maximum value in the image divided
    by the MSR squared (result is in dB). Higher is better.
  * Structure Similarity Index (SSIM), a more advanced metric based on perceptual
    considerations, divided in luminance, contrast and structure. Closer to 1 is better.

* Additionnal information provided:

  * Distance between the detection and the true galaxy
  * Distance to the closest true galaxy
  * Blendedness, defined as:

    .. math::
        1 - \frac{S_k \cdot S_k}{S_{all} \cdot S_k}

    where :math:`S_k` is the flux of the k-th galaxy for each pixel (as a vector),
    :math:`S_{all}` is the flux of all the galaxies for each pixel, and :math:`\cdot`
    is the standard scalar product on vectors.

"""
from abc import ABC, abstractmethod

import numpy as np

from btk.draw_blends import BlendBatch
from btk.measure import MeasuredBatch
from btk.metrics_utils import get_matches


class Metric(ABC):
    """Abstract class for BTK metrics."""

    def __init__(self, batch_size: int, survey_name: str, filter_name: str) -> None:
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
        pass

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
