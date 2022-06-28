"""Toolkit for fast simulation and analysis of overlapping objects for LSSTDESC.

The code generates on the fly images of overlapping parametric galaxies, while
providing a framework to test user defined detection/deblending/measurement
algorithms.
"""
__author__ = "btk developers"
__email__ = "imendoza@umich.edu"
__version__ = "1.0.0a1"

DEFAULT_SEED = 0

from . import (
    catalog,
    create_blend_generator,
    draw_blends,
    measure,
    metrics,
    multiprocess,
    plot_utils,
    sampling_functions,
    survey,
    utils,
)
