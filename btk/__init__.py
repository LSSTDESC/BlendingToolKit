"""Toolkit for fast simulation and analysis of overlapping objects for LSSTDESC.

The code generates on the fly images of overlapping parametric galaxies, while
providing a framework to test user defined detection/deblending/measurement
algorithms.
"""
import importlib.metadata

__author__ = "btk developers"
__email__ = "imendoza@umich.edu"
__version__ = "1.0.0a1"

DEFAULT_SEED = 0

from . import catalog
from . import create_blend_generator
from . import draw_blends
from . import measure
from . import metrics
from . import multiprocess
from . import plot_utils
from . import sampling_functions
from . import survey
from . import utils
