"""Toolkit for fast simulation and analysis of overlapping objects for the
LSST Dark Energy Science Collaboration.

The code generates on the fly images of overlapping parametric galaxies, while
providing a framework to test user defined detection/deblending/measurement
algorithms.
"""

__author__ = "btk developers"
__email__ = "sowmyak@stanford.edu"
__version__ = "0.1"

from . import get_input_catalog
from . import create_blend_generator
from . import create_observing_generator
from . import draw_blends
from . import measure
from . import config
from . import compute_metrics
from . import utils
