import numpy as np


def get_random_center_shift(Args, number_of_objects, maxshift=None):
    """Returns random shifts in x and y coordinates between + and - max-shift
    in arcseconds.

    Args:
        Args: Class containing input parameters.
        number_of_objects(int): Number of x and y shifts to return.
        maxshift (float): Magnitude of maximum value of shift. If None then it
            is set as one-tenth the stamp size.
    """
    if not maxshift:
        maxshift = Args.stamp_size / 10.  # in arcseconds
    dx = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    dy = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    return dx, dy


def default_sampling(Args, catalog):
    """Applies default sampling to the input CatSim-like catalog and returns
    catalog with entries corresponding to a blend centered close to postage
    stamp center.

    Function selects entries from input catalog that are brighter than 25.3 mag
    in the i band. Number of objects per blend is set at a random integer
    between 1 and Args.max_number. The blend catalog is then randomly sampled
    entries from the catalog after selection cuts. The centers are randomly
    distributed within 1/10th of the stamp size. Here even though the galaxies
    are sampled from the CatSim catalog, their spatial location are not
    representative of real blends.

    Args:
        Args: Class containing input parameters.
        catalog: CatSim-like catalog from which to sample galaxies.

    Returns:
        Catalog with entries corresponding to one blend.
    """
    number_of_objects = np.random.randint(1, Args.max_number + 1)
    q, = np.where(catalog['i_ab'] <= 25.3)
    blend_catalog = catalog[np.random.choice(q, size=number_of_objects)]
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    dx, dy = get_random_center_shift(Args, number_of_objects)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def generate(Args, catalog, sampling_function=None):
    """Generates a list of blend catalogs of length Args.batch_size. Each blend
    catalog has entries numbered between 1 and Args.max_number, corresponding
    to overlapping objects in the blend.

    Args:
        Args: Class containing input parameters.
        sampling_function: Function to sample input catalog from which to draw
                           blends.

    Yields:
        Generator for parameters of each galaxy in blend.
    """
    while True:
        blend_catalogs = []
        for i in range(Args.batch_size):
            if sampling_function:
                blend_catalog = sampling_function(Args, catalog)
            else:
                blend_catalog = default_sampling(Args, catalog)
                if Args.verbose:
                    print("Default random sampling of objects from catalog")
            np.testing.assert_array_less(
                len(blend_catalog) - 1, Args.max_number, "Number of objects"
                " per blend must be less than max_number: {0} <= {1}".format(
                    len(blend_catalog), Args.max_number))
            blend_catalogs.append(blend_catalog)
        yield blend_catalogs
