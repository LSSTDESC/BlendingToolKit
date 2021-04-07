from abc import ABC
from abc import abstractmethod


class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(blend_output: dict, measure_output: dict):
        """Operates on one batch at a time using the output from the `MeasureGenerator` class."""

    @abstractmethod
    def from_fits(blend_fits: str, measure_fits: str):
        """Pass in the path to two `.fits` files containing astropy tables of (complete, not in
        batches) true blend information and results of measurement.

        Each table should have a column `blend_id` corresponding to each postage stamp measure,
        to identify which entries should be compared and potentially matched.
        """


class DetectionMetric:
    def __init__(self):
        pass

    def __call__(blend_output, measure_output):
        """Operates on one batch at a time."""
        pass


class SegmentationMetric:
    def __init__(self):
        pass

    def __call__(blend_output: dict, measure_output: dict):
        """Operates on one batch at a time.

        The order of objects in the segmentation must be the same as the order in the `catalog`
        value of the `measure_output` dictionary. The matching calculated using the
        `DetectionMetric` will be used to determine which segmentations to compare.

        Segmentation from `blend_output` can be calculated easily using the value of
        `isolated_images`.
        """
        pass

    @abstractmethod
    def from_fits(blend_fits, measure_fits):
        """In this case, first file contains a separate HUD with the segmentation for each blend?

        Suggestions welcome of how to best structure this.
        """
        pass


def get_metrics(
    cpus=1,
    save_results=None,
    n_batches=None,
    measure_generator=None,
    blend_fits=None,
    measure_fits=None,
    use_metrics=("detection", "segementation"),
):
    """User will only ever need to interact with this function, unless they want to add a new
    metric.

    The metrics themselves are classes to indicate that they require 'heavy' changes and careful
    development.

    Args:
        n_batches (int): How many batches to analyze from the measure_generator?
        measure_generator:
        blend_fits (str): File containing results from draw_blend_generator saved into a
                          `.fits` file.
        measure_fits (str): Results from measurement on `blend_fits` saved into a `.fits` file.

    Returns:
        Results can be returned as an astropy table or saved into a `.fits` files.

    """

    # Need to add utility functions to create `blend_fits` from DrawBlendGenerator.

    assert (measure_generator is not None and n_batches is not None) or (
        blend_fits is not None and measure_fits is not None
    ), "At least one `measure_generator` or pair of `.fits` files is required. "

    # depending on `use_metrics` flag is which metrics will be attempted to be calculated.
    # each metrics class will have an internal check to determine whether it can be calculated
    # from the provided draw_blend_generator or fits files.

    # if using fits files, need to add check that they are well constructed.
