"""For an input catalog of objects and observing conditions draws objects
with WLDeblending package.
ToDo:
Add noise
Add data augmentation(rotation)
fix descwl.render.SourceNotVisible
"""
import descwl
import copy
import galsim
import numpy as np
from astropy.table import Column


def get_center_in_pixels(blend_catalog, obs, Args):
    """Return centroids in pixels"""
    center = (Args.stamp_size/obs.pixel_scale - 1)/2
    dx = blend_catalog['ra']/obs.pixel_scale + center
    dy = blend_catalog['dec']/obs.pixel_scale + center
    dx_col = Column(dx, name='dx')
    dy_col = Column(dy, name='dy')
    return dx_col, dy_col


def generate(Args, blend_genrator, observing_generator):
    """Generates images of blended objects, individual isolated objects,
    PSF image in each batch for each object in the batch

    Args:
        Args: Class containing parameters to create blends
        blend_genrator: Generator to create blended object
        observing_genrator: Creates observing conditions for each entry in
                            batch.
    Returns:
        output: Dictionary with blend image, isolated object image, PSF image,
                mean sky level in each band and, blend catalog per entry per
                batch.
    """
    while True:
        blend_list = next(blend_genrator)
        obs_cond = next(observing_generator)
        stamp_size = np.int(Args.stamp_size / obs_cond[0].pixel_scale)
        blend_images = np.zeros((Args.batch_size, stamp_size, stamp_size,
                                 len(Args.bands)))
        isolated_images = np.zeros((Args.batch_size, Args.max_number,
                                    stamp_size, stamp_size, len(Args.bands)))
        for i in range(Args.batch_size):
            dx, dy = get_center_in_pixels(blend_list[i], obs_cond[0], Args)
            blend_list[i].add_column(dx)
            blend_list[i].add_column(dy)
            blend_list[i].add_column(Column(np.zeros(len(blend_list[i])),
                                            name='not_drawn'))
            for j, band in enumerate(Args.bands):
                blend_obs = copy.deepcopy(obs_cond[j])
                galaxy_builder = descwl.model.GalaxyBuilder(blend_obs, False,
                                                            False, False,
                                                            False)
                blend_render_engine = descwl.render.Engine(
                    survey=blend_obs,
                    min_snr=0.01,
                    truncate_radius=30,
                    no_margin=False,
                    verbose_render=False)
                for k, entry in enumerate(blend_list[i]):
                    try:
                        galaxy = galaxy_builder.from_catalog(entry,
                                                             entry['ra'],
                                                             entry['dec'],
                                                             band)
                        blend_render_engine.render_galaxy(galaxy, True, False)
                        if Args.draw_isolated:
                            if Args.verbose:
                                print("Draw isolated object")
                            iso_obs = copy.deepcopy(obs_cond[j])
                            iso_render_engine = descwl.render.Engine(
                                survey=iso_obs,
                                min_snr=0.01,
                                truncate_radius=30,
                                no_margin=False,
                                verbose_render=False)
                            iso_render_engine.render_galaxy(galaxy, True, False)
                            isolated_images[i, k, :, :, j] = iso_obs.image.array
                    except descwl.render.SourceNotVisible:
                        print("Source not visible")
                        blend_list[i]['not_drawn'][k] = 1
                        continue
                if Args.add_noise:
                    if Args.verbose:
                        print("Noise added to blend image")
                    generator = galsim.random.BaseDeviate(seed=Args.seed)
                    noise = galsim.PoissonNoise(
                        rng=generator,
                        sky_level=blend_obs.mean_sky_level)
                    blend_obs.image.addNoise(noise)
                blend_images[i, :, :, j] = blend_obs.image.array
        output = {'blend_images': blend_images,
                  'isolated_images': isolated_images,
                  'obs_condition': obs_cond,
                  'blend_list': blend_list}
        yield output
