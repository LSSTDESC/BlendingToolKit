"""Contains classes of function for extracing information from catalog in blend batches."""
import warnings
from abc import ABC, abstractmethod

import astropy.table
import numpy as np

from btk import DEFAULT_SEED


def _get_random_center_shift(num_objects, max_shift, rng):
    """Returns random shifts in x and y coordinates between + and - max-shift in arcseconds.

    Args:
        num_objects (int): Number of x and y shifts to return.

    Returns:
        x_peak (float): random shift along the x axis
        y_peak (float): random shift along the x axis
    """
    x_peak = rng.uniform(-max_shift, max_shift, size=num_objects)
    y_peak = rng.uniform(-max_shift, max_shift, size=num_objects)
    return x_peak, y_peak


def _get_random_rotation(num_objects, rng):
    """Returns angle in degrees for random rotation of galaxies.

    Args:
        num_objects (int): Number of ranndom rotation angles to return
        rng (np.random.default_rng): random number generator
    """
    theta = rng.uniform(0, 360, size=num_objects)
    return theta


class SamplingFunction(ABC):
    """Class representing sampling functions to sample input catalog from which to draw blends.

    The object can be called to return an astropy table with entries corresponding to the
    galaxies chosen for the blend.
    """

    def __init__(self, max_number, min_number=1, seed=DEFAULT_SEED, add_rotation=False):
        """Initializes the SamplingFunction.

        Args:
            max_number (int): maximum number of catalog entries returned from sample.
            min_number (int): minimum number of catalog entries returned from sample.
            seed (int): Seed to initialize randomness for reproducibility.
            add_rotation (bool): To chose whether to add random rotations to objects or not.
                If set to True, the sampling fucntion should return a column "theta"
        """
        self.max_number = max_number
        self.min_number = min_number
        self.add_rotation = add_rotation

        if isinstance(seed, int):
            self.rng = np.random.default_rng(seed)
        else:
            raise AttributeError("The seed you provided is invalid, should be an int.")

    @abstractmethod
    def __call__(self, table, **kwargs):
        """Outputs a sample from the given astropy table.

        NOTE: The sample must contain at most self.max_number of objects.
        """

    @property
    @abstractmethod
    def compatible_catalogs(self):
        """Get a tuple of compatible catalogs by their name in ``catalog.py``."""


class DefaultSampling(SamplingFunction):
    """Default sampling function used for producing blend tables."""

    def __init__(
        self,
        max_number=2,
        min_number=1,
        stamp_size=24.0,
        max_shift=None,
        seed=DEFAULT_SEED,
        add_rotation=False,
    ):
        """Initializes default sampling function.

        Args:
            max_number (int): Defined in parent class
            min_number (int): Defined in parent class
            stamp_size (float): Size of the desired stamp.
            max_shift (float): Magnitude of maximum value of shift. If None then it
                             is set as one-tenth the stamp size. (in arcseconds)
            seed (int): Seed to initialize randomness for reproducibility.
            add_rotation (bool): Defined in parent class
        """
        super().__init__(
            max_number=max_number, min_number=min_number, seed=seed, add_rotation=add_rotation
        )
        self.stamp_size = stamp_size
        self.max_shift = max_shift if max_shift else self.stamp_size / 10.0

    @property
    def compatible_catalogs(self):
        """Tuple of compatible catalogs for this sampling function."""
        return "CatsimCatalog", "CosmosCatalog"

    def __call__(self, table, shifts=None, indexes=None):
        """Applies default sampling to the input CatSim-like catalog.

        Returns an astropy table with entries corresponding to a blend centered close to postage
        stamp center.

        Function selects entries from input table that are brighter than 25.3 mag
        in the i band. Number of objects per blend is set at a random integer
        between 1 and ``self.max_number``. The blend table is then randomly sampled
        entries from the table after selection cuts. The centers are randomly
        distributed within 1/10th of the stamp size. Here even though the galaxies
        are sampled from a CatSim catalog, their spatial location are not
        representative of real blends.

        Args:
            table (Astropy.table): Table containing entries corresponding to galaxies
                                   from which to sample.
            shifts (list): Contains arbitrary shifts to be applied instead of random ones.
                           Should of the form [x_peak,y_peak] where x_peak and y_peak are the lists
                           containing the x and y shifts.
            indexes (list): Contains the indexes of the galaxies to use.

        Returns:
            Astropy.table with entries corresponding to one blend.
        """
        number_of_objects = self.rng.integers(self.min_number, self.max_number)
        (q,) = np.where(table["ref_mag"] <= 25.3)

        if indexes is None:
            blend_table = table[self.rng.choice(q, size=number_of_objects)]
        else:
            blend_table = table[indexes]
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0
        if shifts is None:
            x_peak, y_peak = _get_random_center_shift(number_of_objects, self.max_shift, self.rng)
        else:
            x_peak, y_peak = shifts
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak

        blend_table["theta"] = 0.0

        if self.add_rotation:
            blend_table["theta"] = _get_random_rotation(num_objects=number_of_objects, rng=self.rng)

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table


class BasicSampling(SamplingFunction):
    """Example of basic sampling function features.

    Includes magnitude cut, restriction on the shape, shift randomization.
    """

    def __init__(
        self,
        max_number=4,
        min_number=1,
        stamp_size=24.0,
        max_shift=None,
        seed=DEFAULT_SEED,
        add_rotation=False,
    ):
        """Initializes the basic sampling function.

        Args:
            max_number (int): Defined in parent class
            min_number (int): Defined in parent class
            stamp_size (float): Size of the desired stamp.
            max_shift (float): Magnitude of maximum value of shift. If None then it
                             is set as one-tenth the stamp size. (in arcseconds)
            seed (int): Seed to initialize randomness for reproducibility.
            add_rotation (bool): Defined in parent class
        """
        super().__init__(
            max_number=max_number, min_number=min_number, seed=seed, add_rotation=add_rotation
        )
        self.stamp_size = stamp_size
        self.max_shift = max_shift if max_shift else self.stamp_size / 10.0

    @property
    def compatible_catalogs(self):
        """Tuple of compatible catalogs for this sampling function."""
        return ("CatsimCatalog",)

    def __call__(self, table, **kwargs):
        """Samples galaxies from input catalog to make blend scene.

        Then number of galaxies in a blend are drawn from a uniform
        distribution of one up to ``self.max_number``. Function always selects one
        bright galaxy that is less than 24 mag. The other galaxies are selected
        from a sample with i<25.3 90% of the times and the remaining 10% with i<28.
        All galaxies must have semi-major axis is between 0.2 and 2 arcsec.
        The centers are randomly distributed within 1/30 *sqrt(N) of the postage
        stamp size, where N is the number of objects in the blend.

        Args:
            table: CatSim-like catalog from which to sample galaxies.

        Returns:
            Table with entries corresponding to one blend.
        """
        number_of_objects = self.rng.integers(self.min_number, self.max_number)
        a = np.hypot(table["a_d"], table["a_b"])
        cond = (a <= 2) & (a > 0.2)
        (q_bright,) = np.where(cond & (table["ref_mag"] <= 24))
        if self.rng.random() >= 0.9:
            (q,) = np.where(cond & (table["ref_mag"] < 28))
        else:
            (q,) = np.where(cond & (table["ref_mag"] <= 25.3))
        blend_table = astropy.table.vstack(
            [
                table[self.rng.choice(q_bright, size=1)],
                table[self.rng.choice(q, size=number_of_objects)],
            ]
        )
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0

        # keep number density of objects constant
        max_shift = self.stamp_size / 30.0 * number_of_objects**0.5
        x_peak, y_peak = _get_random_center_shift(number_of_objects + 1, max_shift, self.rng)
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak
        if self.add_rotation:
            blend_table["theta"] = _get_random_rotation(num_objects=number_of_objects, rng=self.rng)
        return blend_table
