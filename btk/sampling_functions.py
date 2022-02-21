"""Contains classes of function for extracing information from catalog in blend batches."""
import warnings
from abc import ABC
from abc import abstractmethod

import astropy.table
import numpy as np

from btk import DEFAULT_SEED
from btk.catalog import CatsimCatalog


def _get_random_center_shift(num_objects, maxshift, rng):
    """Returns random shifts in x and y coordinates between + and - max-shift in arcseconds.

    Args:
        num_objects (int): Number of x and y shifts to return.

    Returns:
        x_peak (float): random shift along the x axis
        y_peak (float): random shift along the x axis
    """
    x_peak = rng.uniform(-maxshift, maxshift, size=num_objects)
    y_peak = rng.uniform(-maxshift, maxshift, size=num_objects)
    return x_peak, y_peak


class SamplingFunction(ABC):
    """Class representing sampling functions to sample input catalog from which to draw blends.

    The object can be called to return an astropy table with entries corresponding to the
    galaxies chosen for the blend.
    """

    def __init__(self, max_number, seed=DEFAULT_SEED):
        """Initializes the SamplingFunction.

        Args:
            max_number (int): maximum number of catalog entries returned from sample.
            seed (int): Seed to initialize randomness for reproducibility.
        """
        self.max_number = max_number

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

    def __init__(self, max_number=2, stamp_size=24.0, maxshift=None, seed=DEFAULT_SEED):
        """Initializes default sampling function.

        Args:
            max_number (int): Defined in parent class
            stamp_size (float): Size of the desired stamp.
            maxshift (float): Magnitude of maximum value of shift. If None then it
                             is set as one-tenth the stamp size. (in arcseconds)
            seed (int): Seed to initialize randomness for reproducibility.
        """
        super().__init__(max_number, seed)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0

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
        number_of_objects = self.rng.integers(1, self.max_number + 1)
        (q,) = np.where(table["ref_mag"] <= 25.3)

        if indexes is None:
            blend_table = table[self.rng.choice(q, size=number_of_objects)]
        else:
            blend_table = table[indexes]
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0
        if shifts is None:
            x_peak, y_peak = _get_random_center_shift(number_of_objects, self.maxshift, self.rng)
        else:
            x_peak, y_peak = shifts
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table


class BasicSampling(SamplingFunction):
    """Example of basic sampling function features.

    Includes magnitude cut, restriction on the shape, shift randomization.
    """

    def __init__(self, max_number=4, stamp_size=24.0, maxshift=None, seed=DEFAULT_SEED):
        """Initializes the basic sampling function.

        Args:
            max_number (int): Defined in parent class
            stamp_size (float): Size of the desired stamp.
            maxshift (float): Magnitude of maximum value of shift. If None then it
                             is set as one-tenth the stamp size. (in arcseconds)
            seed (int): Seed to initialize randomness for reproducibility.
        """
        super().__init__(max_number, seed)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0

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
        number_of_objects = self.rng.integers(0, self.max_number)
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
        maxshift = self.stamp_size / 30.0 * number_of_objects**0.5
        x_peak, y_peak = _get_random_center_shift(number_of_objects + 1, maxshift, self.rng)
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak
        return blend_table


class GroupSampling(SamplingFunction):
    """Uses a pre-analyzed WLD catalog to draw blends."""

    def __init__(
        self,
        max_number,
        wld_catalog_name,
        stamp_size,
        pixel_scale,
        shift=None,
        group_id=None,
        seed=DEFAULT_SEED,
    ):
        """Blends are defined from *groups* of galaxies from a CatSim-like catalog.

        Note: the pre-run WLD images are not used here. We only use the pre-run
        catalog (in i band) to identify galaxies that belong to a group.

        Args:
            max_number (int): Same as in SamplingFunction
            wld_catalog_name: File path to a pre-analyzed WLD Catsim catalog
            stamp_size (int): Size of the generated stamps
            pixel_scale (float): pixel scale of the survey, in arcseconds per pixel
            shift (list): List containing shifts to apply (useful to avoid randomization)
            group_id (list): List containing which group_ids to analyze (avoid randomization)
            seed (int): Seed to initialize randomness for reproducibility.
        """
        super().__init__(max_number, seed)

        self.wld_catalog = CatsimCatalog.from_file(wld_catalog_name).get_raw_catalog()
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.shift = shift
        self.group_id = group_id

    @property
    def compatible_catalogs(self):
        """Tuple of compatible catalogs for this sampling function."""
        return ("CatsimCatalog",)

    def __call__(self, table, **kwargs):
        """Retrun blend info based on self.wld_catalog.

        We use self.wld_catalog created above to sample groups, but ultimately returns
        rows from the input ``table`` (by matching the corresponding galaxy ids). The group is
        centered on the middle of the postage stamp. Function only draws galaxies that lie within
        the postage stamp size.
        """
        # randomly sample a group om wld_catalog
        bool_groups = self.wld_catalog["grp_size"] >= 2
        group_ids = np.unique(self.wld_catalog["grp_id"][bool_groups])
        if self.group_id is None:
            group_id = self.rng.choice(group_ids, replace=False)
        else:
            group_id = self.group_id

        # get all galaxies belonging to the group.
        ids = self.wld_catalog["db_id"][self.wld_catalog["grp_id"] == group_id]
        blend_table = astropy.table.vstack([table[table["galtileid"] == i] for i in ids])

        # Set mean x and y coordinates of the group galaxies to the center of the
        # postage stamp.
        blend_table["ra"] -= np.mean(blend_table["ra"])
        blend_table["dec"] -= np.mean(blend_table["dec"])

        # Add small random shift so that center does not perfectly align with
        # the stamp center
        if self.shift is None:
            x_peak, y_peak = _get_random_center_shift(
                1, maxshift=3 * self.pixel_scale, rng=self.rng
            )
        else:
            x_peak, y_peak = self.shift
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak
        # make sure galaxy centers don't lie too close to edge
        cond1 = np.abs(blend_table["ra"]) < self.stamp_size / 2.0 - 3
        cond2 = np.abs(blend_table["dec"]) < self.stamp_size / 2.0 - 3
        no_boundary = blend_table[cond1 & cond2]
        if len(no_boundary) == 0:
            return no_boundary
        # make sure number of galaxies in blend is less than self.max_number
        # randomly select max_number of objects if larger.
        num = min([len(no_boundary), self.max_number])
        select = self.rng.choice(range(len(no_boundary)), num, replace=False)
        return no_boundary[select]


class GroupSamplingNumbered(SamplingFunction):
    """Defines blends based on previously analyzed WLD catalog & exits once all groups are used."""

    def __init__(
        self,
        max_number,
        wld_catalog_name,
        stamp_size,
        pixel_scale,
        shift=None,
        fmt="fits",
        seed=DEFAULT_SEED,
    ):
        """Blends defined from *groups* of galaxies from a catalog previously analyzed with WLD.

        This function has an extra attribute group_id_count which tracks the
        group id returned. Each time the generator is called, 1 gets added to the
        count. If the count is larger than the number of groups input,
        the generator is forced to exit.

        NOTE: the pre-run WLD images are not used here. We only use the pre-run
        catalog to identify galaxies that belong to a group.

        Args:
            max_number: Same as SamplingFunction
            wld_catalog_name: Same as GroupSamplingFunction
            stamp_size (int): Size of the generated stamps
            pixel_scale (float): pixel scale of the survey, in arcseconds per pixel
            fmt (str): Format of input wld_catalog used to define groups.
            shift (list): List of shifts to apply (usefult to avoid randomization)
            seed (int): Seed to initialize randomness for reproducibility.
        """
        super().__init__(max_number, seed)
        self.wld_catalog = astropy.table.Table.read(wld_catalog_name, format=fmt)
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.group_id_count = 0
        self.shift = shift

    @property
    def compatible_catalogs(self):
        """Tuple of compatible catalogs for this sampling function."""
        return ("CatsimCatalog",)

    def __call__(self, table, **kwargs):
        """Return info for one blend from a given astropy table based on groups from wld_catalog.

        The group is centered on the middle of the postage stamp. This function only returns
        galaxies whose centers lie within 1 arcsec the postage stamp edge, which may cause the
        number of galaxies in the blend to be smaller than the group size.
        """
        # randomly sample a group.
        group_ids = np.unique(
            self.wld_catalog["grp_id"][
                (self.wld_catalog["grp_size"] >= 2)
                & (self.wld_catalog["grp_size"] <= self.max_number)
            ]
        )
        if self.group_id_count >= len(group_ids):
            message = "group_id_count is larger than number of groups input"
            raise GeneratorExit(message)

        group_id = group_ids[self.group_id_count]
        self.group_id_count += 1
        # get all galaxies belonging to the group.
        # make sure some group or galaxy was not repeated in wld_catalog
        ids = np.unique(self.wld_catalog["db_id"][self.wld_catalog["grp_id"] == group_id])
        blend_table = astropy.table.vstack([table[table["galtileid"] == i] for i in ids])
        # Set mean x and y coordinates of the group galaxies to the center of the
        # postage stamp.
        blend_table["ra"] -= np.mean(blend_table["ra"])
        blend_table["dec"] -= np.mean(blend_table["dec"])
        # Add small random shift so that center does not perfectly align with stamp
        # center
        if self.shift is None:
            x_peak, y_peak = _get_random_center_shift(
                1, maxshift=5 * self.pixel_scale, rng=self.rng
            )
        else:
            x_peak, y_peak = self.shift
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak
        # make sure galaxy centers don't lie too close to edge
        cond1 = np.abs(blend_table["ra"]) < self.stamp_size / 2.0 - 1
        cond2 = np.abs(blend_table["dec"]) < self.stamp_size / 2.0 - 1
        no_boundary = blend_table[cond1 & cond2]
        message = (
            "Number of galaxies greater than max number of objects per"
            f"blend. Found {len(no_boundary)}, expected <= {self.max_number}"
        )
        assert len(no_boundary) <= self.max_number, message
        return no_boundary
