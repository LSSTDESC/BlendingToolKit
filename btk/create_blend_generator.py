"""The generator creates blend lists according to a given strategy"""
import numpy as np


def get_random_center_shift(number_of_objects, maxshift):
    """Returns a random shift from the center in x and y coordiantes
    between 0 and maxshift.
    """
    dx = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    dy = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    return dx, dy


def random_sample(Args, catalog, number_of_objects):
    """Randomly picks entries from input catlog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    q, = np.where(catalog['i_ab'] <= 25.3)
    blend_catalog = catalog[np.random.choice(q, size=number_of_objects)]
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    maxshift = Args.stamp_size / 10.  # * Args.pixel_scale / 3600.
    dx, dy = get_random_center_shift(number_of_objects, maxshift)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def generate(Args, catalog, sampling_function=None):
    """Generates a list of blend catalogs of length Args.batch_size. Each blend
    catlog has overlapping objects between 1 and Args.max_number.
    """
    while True:
        blend_catalogs = []
        for i in range(Args.batch_size):
            number_of_objects = np.random.randint(1, Args.max_number)
            if sampling_function:
                blend_catalog = sampling_function(Args, catalog,
                                                  number_of_objects)
            else:
                blend_catalog = random_sample(Args, catalog, number_of_objects)
            blend_catalogs.append(blend_catalog)
        yield blend_catalogs
