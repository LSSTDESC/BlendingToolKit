"""Toolkit for fast simulation and analysis of overlapping objects for LSSTDESC.

The code generates on the fly images of overlapping parametric galaxies, while
providing a framework to test user defined detection/deblending/measurement
algorithms.
"""
from importlib import metadata

__author__ = "btk developers"
__email__ = "imendoza@umich.edu"
__version__ = metadata.version("blending_toolkit")

from . import catalog, sampling_functions, survey, multiprocess, match, measure, utils
from . import blend_generator, blend_batch
from . import draw_blends
from . import deblend
from . import metrics
