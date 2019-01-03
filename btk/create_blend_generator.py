"""The generator creates blend lists according to a given strategy"""
import numpy as np


def get_random_center_shift(Args, number_of_objects):
    """Returns a random shift from the center in x and y coordinates
    between 0 and max-shift (in arcseconds).
    """
    maxshift = Args.stamp_size / 10.  # in arcseconds
    dx = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    dy = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    return dx, dy


def random_sample(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    number_of_objects = np.random.randint(1, Args.max_number)
    q, = np.where(catalog['i_ab'] <= 25.3)
    blend_catalog = catalog[np.random.choice(q, size=number_of_objects)]
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    dx, dy = get_random_center_shift(Args, number_of_objects)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def generate(Args, catalog, sampling_function=None):
    """Generates a list of blend catalogs of length Args.batch_size. Each blend
    catalog has overlapping objects between 1 and Args.max_number.
    Args:
        Args: Class containing input parameters.
        sampling_function: Function to sample input catalog from which to draw
                           blends.
    Returns:
        Generator for parameters of each galaxy in blend.
    """
    while True:
        blend_catalogs = []
        for i in range(Args.batch_size):
            if sampling_function:
                blend_catalog = sampling_function(Args, catalog)
            else:
                blend_catalog = random_sample(Args, catalog)
                if Args.verbose:
                    print("Default random sampling of objects from catalog")
            np.testing.assert_array_less(
                len(blend_catalog) - 1, Args.max_number, "Number of objects"
                " per blend must be less than max_number: {0} <= {1}".format(
                    len(blend_catalog), Args.max_number))
            blend_catalogs.append(blend_catalog)
        yield blend_catalogs
