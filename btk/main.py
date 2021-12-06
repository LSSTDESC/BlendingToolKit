"""Implements main function to run BTK end-to-end."""
from collections.abc import Iterable

from hydra.utils import instantiate
from omegaconf import OmegaConf

from btk.measure import available_measure_functions
from btk.measure import MeasureGenerator
from btk.metrics import MetricsGenerator
from btk.survey import get_surveys


def main(cfg: OmegaConf):
    """Run BTK from end-to-end using a hydra configuration object."""
    surveys = get_surveys(list(cfg.surveys))

    # get draw blends generator.
    draw_blend_generator = instantiate(cfg.draw_blends, surveys=surveys)

    # get measure_functions.
    measure_functions = []
    f_names = cfg.measure.measure_functions
    if not isinstance(f_names, Iterable):
        f_names = [f_names]
    for f_name in f_names:
        if f_name not in available_measure_functions:
            raise ValueError("Measure functions specified are not implemented in BTK.")
        measure_functions.append(available_measure_functions[f_name])

    # get measure generator.
    measure_generator = MeasureGenerator(
        measure_functions, draw_blend_generator, **cfg.measure.kwargs
    )

    # get metrics generator
    metrics_generator = MetricsGenerator(measure_generator, **cfg.metrics.kwargs)
    _ = next(metrics_generator)
