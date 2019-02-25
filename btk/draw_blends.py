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
import multiprocessing as mp
from astropy.table import Column
from itertools import chain, starmap


def get_center_in_pixels(blend_catalog, Args):
    """Return centroids in pixels"""
    center = (Args["stamp_size"]/Args["pixel_scale"] - 1)/2
    dx = blend_catalog['ra']/Args["pixel_scale"] + center
    dy = blend_catalog['dec']/Args["pixel_scale"] + center
    dx_col = Column(dx, name='dx')
    dy_col = Column(dy, name='dy')
    return dx_col, dy_col


def draw_isolated(Args, galaxy, iso_obs):
    if Args["verbose"]:
        print("Draw isolated object")
    iso_render_engine = descwl.render.Engine(
        survey=iso_obs,
        min_snr=Args["min_snr"],
        truncate_radius=30,
        no_margin=False,
        verbose_render=False)
    iso_render_engine.render_galaxy(
        galaxy, no_partials=True, calculate_bias=False, no_analysis=True)
    return iso_obs.image.array


def run_single_band(Args, blend_cat,
                    obs_cond,
                    band):
    blend_obs = copy.deepcopy(obs_cond)
    iso_obs = copy.deepcopy(obs_cond)
    blend_cat.add_column(Column(np.zeros(len(blend_cat)),
                                name='not_drawn_' + band))
    galaxy_builder = descwl.model.GalaxyBuilder(
        blend_obs, no_disk=False, no_bulge=False,
        no_agn=False, verbose_model=False)
    blend_render_engine = descwl.render.Engine(
        survey=blend_obs,
        min_snr=Args["min_snr"],
        truncate_radius=30,
        no_margin=False,
        verbose_render=False)
    stamp_size = np.int(Args["stamp_size"] / Args["pixel_scale"])
    iso_image = np.zeros(
        (Args["max_number"], stamp_size, stamp_size))
    for k, entry in enumerate(blend_cat):
        try:
            galaxy = galaxy_builder.from_catalog(entry,
                                                 entry['ra'],
                                                 entry['dec'],
                                                 band)
            blend_render_engine.render_galaxy(
                galaxy, no_partials=True, calculate_bias=False,
                no_analysis=True)
            if Args["draw_isolated"]:
                iso_image[k] = draw_isolated(Args, galaxy, iso_obs)
        except descwl.render.SourceNotVisible:
            print("Source not visible")
            blend_cat['not_drawn_' + band][k] = 1
            continue
    if Args["add_noise"]:
        if Args["verbose"]:
            print("Noise added to blend image")
        generator = galsim.random.BaseDeviate(seed=np.random.randint(99999999))
        noise = galsim.PoissonNoise(
            rng=generator,
            sky_level=blend_obs.mean_sky_level)
        blend_obs.image.addNoise(noise)
    blend_image = blend_obs.image.array
    return blend_image, iso_image


def single_blend(Args, blend_list, obs_cond, index):
    # print("start entry", index, time.time())
    # start = time.time()
    batch_blend_outputs = []
    for i in range(len(blend_list)):
        dx, dy = get_center_in_pixels(blend_list[i], Args)
        blend_list[i].add_column(dx)
        blend_list[i].add_column(dy)
        stamp_size = np.int(Args["stamp_size"] / Args["pixel_scale"])
        iso_image_multi = np.zeros(
                    (Args["max_number"], stamp_size, stamp_size, len(Args["bands"])))
        blend_image_multi = np.zeros(
                    (stamp_size, stamp_size, len(Args["bands"])))
        for j in range(len(Args["bands"])):
            single_band_output = run_single_band(Args, blend_list[i],
                                                 obs_cond[j], Args["bands"][j])
            blend_image_multi[:, :, j] = single_band_output[0]
            iso_image_multi[:, :, :, j] = single_band_output[1]
        single_output = [blend_image_multi, iso_image_multi,
                         blend_list[i], obs_cond]
        batch_blend_outputs.append(single_output)
    # print("end entry", index, time.time())
    # print("Time taken to finish ", index, time.time()-start)
    return batch_blend_outputs


def generate(Args, blend_genrator, observing_generator,
             multiprocessing=False, cpu=1):
    """Generates images of blended objects, individual isolated objects,
    PSF image in each batch for each object in the batch

    Args:
        Args: Class containing parameters to create blends
        blend_genrator: Generator to create blended object
        observing_genrator: Creates observing conditions for each entry in
                            batch.
        multiprocessing: Divides batch of blends to draw into mini-batches and\
                         runs each on different core
    Returns:
        output: Dictionary with blend images, isolated object images, observing
        conditions in each band and, blend catalog per blend per
        batch.
    """
    while True:
        batch_blend_cat, batch_obs_cond = [], []
        stamp_size = np.int(Args.stamp_size / Args.pixel_scale)
        blend_images = np.zeros((Args.batch_size, stamp_size, stamp_size,
                                 len(Args.bands)))
        isolated_images = np.zeros((Args.batch_size, Args.max_number,
                                    stamp_size, stamp_size, len(Args.bands)))
        in_batch_blend_cat = next(blend_genrator)
        obs_cond = next(observing_generator)
        mini_batch_size = Args.batch_size//cpu
        in_args = [(Args.__dict__, in_batch_blend_cat[i:i+mini_batch_size],
                    copy.deepcopy(obs_cond), i) for i in range(
                        0, Args.batch_size, mini_batch_size)]
        if multiprocessing:
            if Args.verbose:
                print(f"Running mini-batch of size {len(in_args)} \
                    with multiprocessing with pool ", cpu)
            with mp.Pool(processes=cpu) as pool:
                single_blend_results = pool.starmap(single_blend, in_args)
        else:
            if Args.verbose:
                print(f"Running mini-batch of size {len(in_args)} \
                    serial {cpu} times")
            single_blend_results = list(starmap(single_blend, in_args))
        single_blend_results = list(chain(*single_blend_results))
        for i in range(Args.batch_size):
            blend_images[i] = single_blend_results[i][0]
            isolated_images[i] = single_blend_results[i][1]
            batch_blend_cat.append(single_blend_results[i][2])
            batch_obs_cond.append(single_blend_results[i][3])
        output = {'blend_images': blend_images,
                  'isolated_images': isolated_images,
                  'blend_list': batch_blend_cat,
                  'obs_condition': batch_obs_cond}
        yield output
