import os
import numpy as np
import sys
import pytest
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)
import btk
import btk.config


def get_draw_generator(batch_size=8, cpus=1, multiprocessing=False):
    catalog_name = os.path.join(parentdir, 'OneDegSq.fits')
    param = btk.config.Simulation_params(catalog_name, max_number=2,
                                         batch_size=batch_size,
                                         verbose=False,
                                         add_noise=False)
    np.random.seed(param.seed)
    catalog = btk.get_input_catalog.load_catalog(param)
    blend_generator = btk.create_blend_generator.generate(param, catalog)
    observing_generator = btk.create_observing_generator.generate(param)
    draw_generator = btk.draw_blends.generate(param, blend_generator,
                                              observing_generator,
                                              multiprocessing=multiprocessing,
                                              cpus=cpus)
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


@pytest.mark.timeout(5)
def test_draw():
    default_draw_generator = get_draw_generator()
    draw_output = next(default_draw_generator)
    assert len(draw_output['blend_list']) == 8, "Default batch should return 8"
    assert len(draw_output['blend_list'][3]) < 3, "Default max_number should \
        generate 2 or 1 galaxies per blend."
    assert draw_output['obs_condition'][5][0].survey_name == 'LSST', "Default \
        observing survey is LSST."

    pass
