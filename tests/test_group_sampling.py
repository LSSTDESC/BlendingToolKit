import numpy as np

import btk.catalog
import btk.draw_blends
import btk.sampling_functions
import btk.survey


def get_group_sampling_draw_generator(batch_size=3):
    """Returns draw generator with group sampling function"""
    wld_catalog_name = "data/sample_group_catalog.fits"
    catalog_name = "data/sample_group_input_catalog.fits"

    max_number = 10
    stamp_size = 24
    survey = btk.survey.Rubin
    pixel_scale = 0.2
    shift = [0.8, -0.7]
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    sampling_function = btk.sampling_functions.GroupSamplingFunctionNumbered(
        max_number, wld_catalog_name, stamp_size, pixel_scale, shift=shift
    )
    draw_blend_generator = btk.draw_blends.CatsimGenerator(
        catalog, sampling_function, [survey], batch_size=batch_size
    )
    return draw_blend_generator


def test_group_sampling():
    """Test blends drawn with group sampling function"""
    draw_blend_generator = get_group_sampling_draw_generator()
    output = next(draw_blend_generator)
    blend_images = output["blend_images"]
    batch_max = blend_images.max(axis=(0, 2, 3))
    batch_mean = blend_images.mean()
    batch_std = blend_images.std()
    test_batch_max = np.array([17e3, 30e3, 45e3, 43e3, 13e3, 13e2])
    test_batch_mean = 82.1
    test_batch_std = 1027.6

    rel_diff1 = (test_batch_max - batch_max) / test_batch_max
    rel_diff2 = (batch_mean - test_batch_mean) / test_batch_mean
    rel_diff3 = (batch_std - test_batch_std) / test_batch_std
    assert np.all(rel_diff1 <= 0.1)
    assert rel_diff2 <= 0.1
    assert rel_diff3 <= 0.1
