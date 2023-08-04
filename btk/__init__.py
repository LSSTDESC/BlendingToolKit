"""Toolkit for fast simulation and analysis of overlapping objects for LSSTDESC.

The code generates on the fly images of overlapping parametric galaxies, while
providing a framework to test user defined detection/deblending/measurement
algorithms.
"""
from importlib import metadata

__author__ = "btk developers"
__email__ = "imendoza@umich.edu"
__version__ = metadata.version("blending_toolkit")

DEFAULT_SEED = 0

from . import (
    blend_generator,
    catalog,
    deblend,
    draw_blends,
    match,
    measure,
    metrics,
    multiprocess,
    sampling_functions,
    survey,
    utils,
)
