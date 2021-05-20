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
    sampling_function = available_sampling_functions[cfg.sampling.name](**cfg.sampling.kwargs)

    # get survey(s) to be used.
    if not isinstance(cfg.surveys, Iterable):
        cfg.surveys = [cfg.surveys]

    surveys = []
    for survey_name in cfg.surveys:
        if survey_name not in available_surveys:
            raise ValueError(f"Survey '{survey_name}' is not implemented in BTK.")
        survey = available_surveys[survey_name]
        surveys.append(survey)
        # TODO: Possibility to customize PSF inside each filter, etc.

    # get draw blends generator.
    if cfg.draw_blends.name not in available_draw_blends:
        raise ValueError("DrawBlendGenerator specified is not implemented in BTK.")
    draw_blend_generator = available_draw_blends[cfg.draw_blends.name](
        catalog, sampling_function, surveys, **cfg.draw_blends.kwargs
    )

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
