import os
import numpy as np
import sys
import ipdb
import pytest
# import cProfile
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)
import btk
import btk.config


def get_draw_generator(batch_size, cpus, multiprocessing=False):
    catalog_name = os.path.join(parentdir, 'OneDegSq.fits')
    param = btk.config.Simulation_params(catalog_name, max_number=2,
                                         batch_size=batch_size,
                                         verbose=False,
                                         draw_isolated=True,
                                         add_noise=False)
    np.random.seed(param.seed)
    catalog = btk.get_input_catalog.load_catalog(param)
    blend_generator = btk.create_blend_generator.generate(param, catalog)
    observing_generator = btk.create_observing_generator.generate(param)
    draw_generator = btk.draw_blends.generate(param, blend_generator,
                                              observing_generator,
                                              multiprocessing=multiprocessing,
                                              cpu=cpus)
    return draw_generator


@pytest.mark.timeout(15)
def test_multi_processing():
    b_size = 16
    cpus = os.cpu_count()
    parallel_im_gen = get_draw_generator(b_size, cpus, multiprocessing=True)
    parallel_im = next(parallel_im_gen)
    serial_im_gen = get_draw_generator(b_size, cpus, multiprocessing=False)
    serial_im = next(serial_im_gen)
    np.testing.assert_array_equal(parallel_im['blend_images'],
                                  serial_im['blend_images'])
    np.testing.assert_array_equal(parallel_im['isolated_images'],
                                  serial_im['isolated_images'])
    pass
