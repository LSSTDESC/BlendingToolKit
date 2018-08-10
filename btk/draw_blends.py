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


def generate(Args, blend_genrator, observing_genrator):
    while True:
        blend_list = next(blend_genrator)
        obs_cond = next(observing_genrator)
        stamp_size = np.int(Args.stamp_size / obs_cond[0].pixel_scale)
        blend_images = np.zeros((Args.batch_size, stamp_size, stamp_size,
                                 len(Args.bands)))
        isolated_images = np.zeros((Args.batch_size, Args.max_number,
                                    stamp_size, stamp_size, len(Args.bands)))
        for i in range(Args.batch_size):
            for j, band in enumerate(Args.bands):
                blend_obs = copy.deepcopy(obs_cond[j])
                galaxy_builder = descwl.model.GalaxyBuilder(blend_obs, False,
                                                            False, False,
                                                            False)
                blend_render_engine = descwl.render.Engine(survey=blend_obs,
                                                           min_snr=0.05,
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
                        iso_obs = copy.deepcopy(obs_cond[j])
                        iso_render_engine = descwl.render.Engine(survey=iso_obs,
                                                                 min_snr=0.05,
                                                                 truncate_radius=30,
                                                                 no_margin=False,
                                                                 verbose_render=False)
                        iso_render_engine.render_galaxy(galaxy, True, False)
                        isolated_images[i, k, :, :, j] = iso_obs.image.array
                    except descwl.render.SourceNotVisible:
                        print("Source not visible")
                        continue
                if Args.add_noise:
                    generator = galsim.random.BaseDeviate(seed=Args.seed)
                    noise = galsim.PoissonNoise(rng=generator,
                                                sky_level=blend_obs.mean_sky_level)
                    blend_obs.image.addNoise(noise)
                blend_images[i, :, :, j] = blend_obs.image.array
        yield blend_images, isolated_images, blend_list
