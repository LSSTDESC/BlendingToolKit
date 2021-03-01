%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import btk
import btk.plot_utils
import btk.survey
import btk.draw_blends
import astropy.table

stamp_size = 24.0 # arcsecs
catalog_name = "../data/sample_input_catalog.fits"
catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
sampling_function = btk.sampling_functions.DefaultSampling(stamp_size=stamp_size, max_number=3)
draw_generator = btk.draw_blends.CatsimGenerator(
    catalog,
    sampling_function,
    [btk.survey.Rubin],
    batch_size=8,
    stamp_size=stamp_size,
    shifts=None,
    indexes=None,
    multiprocessing=False,
    cpus=1,
    add_noise=True,
)

batch = next(draw_generator)
blend_images = batch['blend_images']
blend_list = batch['blend_list']
btk.plot_utils.plot_blends(blend_images, blend_list, limits=(30,90))