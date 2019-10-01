import descwl
import copy
import galsim
import numpy as np
import multiprocessing as mp
from astropy.table import Column
from itertools import chain, starmap


def get_center_in_pixels(Args, blend_catalog):
    """Returns center of objects in blend_catalog in pixel coordinates of
    postage stamp.

    blend_catalog contains ra dec of object center with the postage stamp
    center being 0,0. The size of the postage stamp and pixel scale is used to
    compute the object centers in pixel coordinates. Coordinates are in pixels
    where bottom left corner of postage stamp is (0, 0).

    Args:
        Args: Class containing input parameters.
        blend_catalog: Catalog with entries corresponding to one blend.

    Returns:
        `astropy.table.Column`s: x and y coordinates of object centroid
    """
    center = (Args.stamp_size/Args.pixel_scale - 1)/2
    dx = blend_catalog['ra']/Args.pixel_scale + center
    dy = blend_catalog['dec']/Args.pixel_scale + center
    dx_col = Column(dx, name='dx')
    dy_col = Column(dy, name='dy')
    return dx_col, dy_col


def get_size(Args, catalog, i_obs_cond):
    """Returns a astropy.table.column with the size of the galaxy.

    Galaxy size is estimated as second moments size (r_sec) computed as
    described in A1 of Chang et.al 2012. The PSF second moment size, psf_r_sec,
    is computed by galsim from the psf model in obs_cond in the i band.
    The object size is the defined as sqrt(r_sec**2 + 2*psf_r_sec**2).

    Args:
        Args: Class containing input parameters.
        catalog: Catalog with entries corresponding to one blend.
        i_obs_cond: `descwl.survey.Survey` class describing
            observing conditions in i band.

    Returns:
        `astropy.table.Column`s: size of the galaxy.
    """
    f = catalog['fluxnorm_bulge']/(catalog['fluxnorm_disk']+catalog['fluxnorm_bulge'])
    hlr_d = np.sqrt(catalog['a_d']*catalog['b_d'])
    hlr_b = np.sqrt(catalog['a_b']*catalog['b_b'])
    r_sec = np.hypot(hlr_d*(1-f)**0.5*4.66,
                     hlr_b*f**0.5*1.46)
    psf = i_obs_cond.psf_model
    psf_r_sec = psf.calculateMomentRadius()
    size = np.sqrt(r_sec**2 + psf_r_sec**2) / Args.pixel_scale
    return Column(size, name='size')


def draw_isolated(Args, galaxy, iso_obs):
    """Returns `descwl.survey.Survey` class object that includes the rendered
    object for an isolated galaxy.

    Args:
        Args: Class containing input parameters.
        galaxy: `descwl.model.Galaxy` class that models galaxies.
        iso_obs: `descwl.survey.Survey` class describing observing conditions.
            The input galaxy is rendered and stored in here.
    """
    if Args.verbose:
        print("Draw isolated object")
    iso_render_engine = descwl.render.Engine(
        survey=iso_obs,
        min_snr=Args.min_snr,
        truncate_radius=30,
        no_margin=False,
        verbose_render=False)
    iso_render_engine.render_galaxy(
        galaxy, variations_x=None, variations_s=None, variations_g=None,
        no_fisher=True, calculate_bias=False, no_analysis=True)
    return iso_obs


def run_single_band(Args, blend_catalog,
                    obs_cond, band):
    """Draws image of isolated galaxies along with the blend image in the
    single input band.

    The WLDeblending package (descwl) renders galaxies corresponding to the
    blend_catalog entries and with observing conditions determined by
    obs_cond. The rendered objects are stored in the the observing conditions
    class. So as to not overwrite images across different blends, we make a
    copies of the obs_cond while drawing each galaxy. Images of isolated
    galaxies are drawn with the WLDeblending and them summed to produce the
    blend image.

    A column 'not_drawn_{band}' is added to blend_catalog initialized as zero.
    If a galaxy was not drawn by descwl, then this flag is set to 1.
    Args:
        Args: Class containing input parameters.
        blend_catalog: Catalog with entries corresponding to one blend.
        obs_cond: `descwl.survey.Survey` class describing observing conditions.
        band(string): Name of band to draw images in.

    Returns:
        Images of blend and isolated galaxies as `numpy.ndarray`.

    """
    blend_catalog.add_column(Column(np.zeros(len(blend_catalog)),
                             name='not_drawn_' + band))
    galaxy_builder = descwl.model.GalaxyBuilder(
        obs_cond, no_disk=False, no_bulge=False,
        no_agn=False, verbose_model=False)
    stamp_size = np.int(Args.stamp_size / Args.pixel_scale)
    iso_image = np.zeros(
        (Args.max_number, stamp_size, stamp_size))
    # define temporary galsim image to hold isolated galaxy images that will be summed
    blend_image_temp = galsim.Image(np.zeros((stamp_size, stamp_size)))
    for k, entry in enumerate(blend_catalog):
        iso_obs = copy.deepcopy(obs_cond)
        try:
            galaxy = galaxy_builder.from_catalog(entry,
                                                 entry['ra'],
                                                 entry['dec'],
                                                 band)
            iso_render = draw_isolated(Args, galaxy, iso_obs)
            iso_image[k] = iso_render.image.array
            blend_image_temp += iso_render.image
        except descwl.render.SourceNotVisible:
            if Args.verbose:
                print("Source not visible")
            blend_catalog['not_drawn_' + band][k] = 1
            continue
        if Args.add_noise:
            if Args.verbose:
                print("Noise added to blend image")
            generator = galsim.random.BaseDeviate(
                seed=np.random.randint(99999999))
            noise = galsim.PoissonNoise(
                rng=generator,
                sky_level=iso_obs.mean_sky_level)
            blend_image_temp.addNoise(noise)
    blend_image = blend_image_temp.array
    return blend_image, iso_image


def run_mini_batch(Args, blend_list, obs_cond):
    """Returns isolated and blended images for bend catalogs in blend_list


    Function loops over blend_list and draws blend and isolated images in each
    band. Even though blend_list was input to the function, we return it since,
    the blend catalogs now include additional columns that flag if an object
    was not drawn and object centers in pixel coordinates.

    Args:
        Args: Class containing input parameters.
        blend_list: List of catalogs with entries corresponding to one blend.
        obs_cond (list): List of `descwl.survey.Survey` class describing
            observing conditions in differnt bands.

    Returns:
        `numpy.ndarray` of blend images and isolated galaxy images, along with
        list of blend catalogs.
    """
    mini_batch_outputs = []
    for i in range(len(blend_list)):
        dx, dy = get_center_in_pixels(Args, blend_list[i])
        blend_list[i].add_column(dx)
        blend_list[i].add_column(dy)
        size = get_size(Args, blend_list[i], obs_cond[3])
        blend_list[i].add_column(size)
        stamp_size = np.int(Args.stamp_size / Args.pixel_scale)
        iso_image_multi = np.zeros(
            (Args.max_number, stamp_size, stamp_size,
             len(Args.bands)))
        blend_image_multi = np.zeros(
                    (stamp_size, stamp_size, len(Args.bands)))
        for j in range(len(Args.bands)):
            single_band_output = run_single_band(Args, blend_list[i],
                                                 obs_cond[j], Args.bands[j])
            blend_image_multi[:, :, j] = single_band_output[0]
            iso_image_multi[:, :, :, j] = single_band_output[1]
        mini_batch_outputs.append([blend_image_multi, iso_image_multi,
                                   blend_list[i]])
    return mini_batch_outputs


def generate(Args, blend_genrator, observing_generator,
             multiprocessing=False, cpus=1):
    """Generates images of blended objects, individual isolated objects, for
    each blend in the batch.


    Batch is divided into mini batches of size Args.batch_size//cpus and each
    mini-batch analyzed separately. The results are then combined to output a
    dict with results of entire batch. If multiprocessing is true, then each of
    the mini-batches are run in parallel.

    Args:
        Args: Class containing parameters to create blends
        blend_genrator: Generator to create blended object
        observing_genrator: Creates observing conditions for each entry in
            batch.
        multiprocessing: Divides batch of blends to draw into mini-batches and
            runs each on different core
        cpus: If multiprocessing, then number of parallel processes to run.

    Yields:
        Dictionary with blend images, isolated object images, blend catalog,
        and observing conditions.

    To do:
        Add data augmentation.
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
        mini_batch_size = Args.batch_size//cpus
        in_args = [(Args, in_batch_blend_cat[i:i+mini_batch_size],
                    copy.deepcopy(obs_cond)) for i in range(
                        0, Args.batch_size, mini_batch_size)]
        if multiprocessing:
            if Args.verbose:
                print("Running mini-batch of size {0} \
                    with multiprocessing with pool {1}".format(
                        len(in_args), cpus))
            with mp.Pool(processes=cpus) as pool:
                mini_batch_results = pool.starmap(run_mini_batch, in_args)
        else:
            if Args.verbose:
                print("Running mini-batch of size {0} \
                    serial {1} times".format(len(in_args), cpus))
            mini_batch_results = list(starmap(run_mini_batch, in_args))
        batch_results = list(chain(*mini_batch_results))
        for i in range(Args.batch_size):
            blend_images[i] = batch_results[i][0]
            isolated_images[i] = batch_results[i][1]
            batch_blend_cat.append(batch_results[i][2])
            batch_obs_cond.append(obs_cond)
        output = {'blend_images': blend_images,
                  'isolated_images': isolated_images,
                  'blend_list': batch_blend_cat,
                  'obs_condition': batch_obs_cond}
        yield output
