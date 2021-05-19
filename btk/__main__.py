"""Implements command-line interface for BTK."""

from collections.abc import Iterable
import hydra
from omegaconf import OmegaConf

from btk.catalog import available_catalogs
from btk.survey import available_surveys
from btk.sampling_functions import available_sampling_functions
from btk.draw_blends import available_draw_blends
from btk.measure import available_measure_functions, MeasureGenerator
from btk.metrics import MetricsGenerator


@hydra.main(config_path="../config", config_name="config")
def main(cfg: OmegaConf):
    # get catalog
    if cfg.catalog.name not in available_catalogs:
        raise ValueError(f"Catalog '{cfg.catalog.name}' is not implemented in BTK.")
    catalog = available_catalogs[cfg.catalog.name].from_file(cfg.catalog.catalog_files)

    # get sampling function
    if cfg.sampling.name not in available_sampling_functions:
        raise ValueError(f"Sampling function '{cfg.sampling.name}' is not implemented in BTK.")
    sampling_function = available_measure_functions[cfg.sampling.name]

    # get survey(s) to be used.
    if not isinstance(cfg.surveys, Iterable):
        cfg.surveys = [cfg.surveys]

    surveys = []
    for survey in cfg.surveys:
        if survey.name not in available_surveys:
            raise ValueError(f"Survey '{survey.name}' is not implemented in BTK.")
        s = available_surveys[survey.name]

        # only use the filters specified.
        s.filters = [filt for filt in s.filters if filt.name in survey.filters]

        # TODO: Possibility to customize PSF inside each filter.

        surveys.append(s)

    # get draw blends generator.
    if cfg.draw_blend.name not in available_draw_blends:
        raise ValueError("DrawBlendGenerator specified is not implemented in BTK.")
    draw_blend_generator = available_draw_blends[cfg.draw_blend.name](
        catalog, sampling_function, surveys, **cfg.draw_blends.kwargs
    )

    # get measure_functions.
    measure_functions = cfg.measure.measure_functions
    if not isinstance(measure_functions, Iterable):
        measure_functions = [cfg.measure.measure_functions]
    for i, f in enumerate(measure_functions):
        if f not in available_measure_functions:
            raise ValueError("Measure functions specified are not implemented in BTK.")
        measure_functions[i] = available_measure_functions[f]

    # get measure generator.
    measure_generator = MeasureGenerator(
        measure_functions, draw_blend_generator, **cfg.measure.kwargs
    )

    # get metrics generator
    metrics_generator = MetricsGenerator(measure_generator, **cfg.metrics.kwargs)
    _ = next(metrics_generator)
