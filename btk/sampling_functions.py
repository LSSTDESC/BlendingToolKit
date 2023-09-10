"""Contains classes of function for extracing information from catalog in blend batches."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import astropy
import numpy as np
from astropy.table import Table

from btk.utils import DEFAULT_SEED


def _get_random_center_shift(
    num_objects: int, max_shift: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns random shifts in x and y coordinates between + and - max-shift in arcseconds.

    Args:
        num_objects: Number of x and y shifts to return.
        max_shift: Maximum value of shift in arcseconds.
        rng: Random number generator.

    Returns:
        dx: random shift along the x axis
        dy: random shift along the x axis
    """
    dx = rng.uniform(-max_shift, max_shift, size=num_objects)
    dy = rng.uniform(-max_shift, max_shift, size=num_objects)
    return dx, dy


def _check_centroids_in_bounds(ra: np.ndarray, dec: np.ndarray, stamp_size: float) -> bool:
    """Checks if the centroids are within the stamp.

    Args:
        ra: Right ascension of centroids in arcseconds.
        dec: Declination of centroids in arcseconds.
        stamp_size: Size of the stamp in arcseconds.

    Returns:
        True if centroids are within the stamp, False otherwise.
    """
    return np.all(np.abs(ra) <= stamp_size / 2.0) and np.all(np.abs(dec) <= stamp_size / 2.0)


def _raise_error_if_out_of_bounds(ra: np.ndarray, dec: np.ndarray, stamp_size: float):
    """Raises ValueError if the centroids are outside the stamp.

    Args:
        ra: Right ascension of centroids in arcseconds.
        dec: Declination of centroids in arcseconds.
        stamp_size: Size of the stamp in arcseconds.
    """
    if not _check_centroids_in_bounds(ra, dec, stamp_size):
        raise ValueError("Object center lies outside the stamp")


class SamplingFunction(ABC):
    """Class representing sampling functions to sample input catalog from which to draw blends.

    The object can be called to return an astropy table with entries corresponding to the
    galaxies chosen for the blend.
    """

    def __init__(self, max_number: int, min_number: int = 1, seed=DEFAULT_SEED):
        """Initializes the SamplingFunction.

        Args:
            max_number: maximum number of catalog entries returned from sample.
            min_number: minimum number of catalog entries returned from sample. (Default: 1)
            seed: Seed to initialize randomness for reproducibility. (Default: btk.DEFAULT_SEED)
        """
        self.min_number = min_number
        self.max_number = max_number

        if self.min_number > self.max_number:
            raise ValueError("Need to satisfy: `min_number <= max_number`")

        if isinstance(seed, int):
            self.rng = np.random.default_rng(seed)
        else:
            raise AttributeError("The seed you provided is invalid, should be an int.")

    @abstractmethod
    def __call__(self, table) -> Table:
        """Outputs a sample from the given astropy table."""


class DefaultSampling(SamplingFunction):
    """Default sampling function used for producing blend catalogs."""

    def __init__(
        self,
        max_number: int = 2,
        min_number: int = 1,
        stamp_size: float = 24.0,
        max_shift: Optional[float] = None,
        seed: int = DEFAULT_SEED,
        max_mag: float = 25.3,
        min_mag: float = -np.inf,
        mag_name: str = "i_ab",
    ):
        """Initializes default sampling function.

        Args:
            max_number: Defined in parent class
            min_number: Defined in parent class
            stamp_size: Size of the desired stamp.
            max_shift: Magnitude of maximum value of shift. If None then it
                             is set as one-tenth the stamp size. (in arcseconds)
            seed: Seed to initialize randomness for reproducibility.
            min_mag: Minimum magnitude allowed in samples
            max_mag: Maximum magnitude allowed in samples.
            mag_name: Name of the magnitude column in the catalog.
        """
        super().__init__(max_number=max_number, min_number=min_number, seed=seed)
        self.stamp_size = stamp_size
        self.max_shift = max_shift if max_shift is not None else self.stamp_size / 10.0
        self.min_mag, self.max_mag = min_mag, max_mag
        self.mag_name = mag_name

    def __call__(self, table: Table) -> Table:
        """Applies default sampling to catalog.

        Returns an astropy table with entries corresponding to a blend centered close to postage
        stamp center.

        Number of objects per blend is set at a random integer between ``self.min_number``
        and ``self.max_number``. The blend table is then randomly sampled entries
        from the table after magnitude selection cuts. The centers are randomly
        distributed within ``self.max_shift`` of the center of the postage stamp.

        Here even though the galaxies are sampled from a CatSim catalog, their spatial
        location are not representative of real blends.

        Args:
            table: Table containing entries corresponding to galaxies
                                    from which to sample.

        Returns:
            Astropy.table with entries corresponding to one blend.
        """
        if self.mag_name not in table.colnames:
            raise ValueError(f"Catalog must have '{self.mag_name}' column.")

        number_of_objects = self.rng.integers(self.min_number, self.max_number + 1)

        cond = (table[self.mag_name] <= self.max_mag) & (table[self.mag_name] > self.min_mag)
        (q,) = np.where(cond)
        blend_table = table[self.rng.choice(q, size=number_of_objects)]

        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0
        dx, dy = _get_random_center_shift(number_of_objects, self.max_shift, self.rng)
        blend_table["ra"] += dx
        blend_table["dec"] += dy
        _raise_error_if_out_of_bounds(blend_table["ra"], blend_table["dec"], self.stamp_size)
        return blend_table


class BasicSampling(SamplingFunction):
    """Example of basic sampling function features.

    Includes magnitude cut, restriction on the shape, shift randomization.
    """

    def __init__(
        self,
        max_number: int = 4,
        min_number: int = 1,
        stamp_size: float = 24.0,
        mag_name: str = "i_ab",
        seed: int = DEFAULT_SEED,
    ):
        """Initializes the basic sampling function.

        Args:
            max_number: Defined in parent class.
            min_number: Defined in parent class.
            stamp_size: Size of the desired stamp.
            seed: Seed to initialize randomness for reproducibility.
            mag_name: Name of the magnitude column in the catalog for cuts.
        """
        super().__init__(max_number=max_number, min_number=min_number, seed=seed)
        self.stamp_size = stamp_size
        self.mag_name = mag_name

        if min_number < 1:
            raise ValueError("At least 1 bright galaxy will be added, so need min_number >=1.")

    def __call__(self, table: Table) -> Table:
        """Samples galaxies from input catalog to make blend scene.

        Then number of galaxies in a blend are drawn from a uniform distribution of one
        up to ``self.max_number``.

        Function always selects one bright galaxy that is less than 24 mag. The other
        galaxies are selected from a sample with mag<25.3 90% of the times and the
        remaining 10% with mag<28.

        All galaxies must have semi-major axis is between 0.2 and 2 arcsec.

        The centers are randomly distributed within 1/30 * sqrt(N) of the postage
        stamp size, where N is the number of objects in the blend. (keeps density constant)

        Args:
            table: CatSim-like catalog from which to sample galaxies.

        Returns:
            Table with entries corresponding to one blend.
        """
        if self.mag_name not in table.colnames:
            raise ValueError(f"Catalog must have '{self.mag_name}' column.")
        if "a_d" not in table.colnames or "a_b" not in table.colnames:
            raise ValueError("Catalog must have 'a_d' and 'a_b' columns.")

        number_of_objects = self.rng.integers(self.min_number - 1, self.max_number)
        a = np.hypot(table["a_d"], table["a_b"])
        cond = (a <= 2) & (a > 0.2)
        (q_bright,) = np.where(cond & (table[self.mag_name] <= 24))
        if self.rng.random() >= 0.9:
            (q,) = np.where(cond & (table[self.mag_name] < 28))
        else:
            (q,) = np.where(cond & (table[self.mag_name] <= 25.3))
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
        dx, dy = _get_random_center_shift(number_of_objects + 1, max_shift, self.rng)
        blend_table["ra"] += dx
        blend_table["dec"] += dy

        _raise_error_if_out_of_bounds(blend_table["ra"], blend_table["dec"], self.stamp_size)
        return blend_table


class DefaultSamplingShear(DefaultSampling):
    """Same as `DefaultSampling` sampling function but includes shear."""

    def __init__(
        self,
        max_number: int = 2,
        min_number: int = 1,
        stamp_size: float = 24.0,
        max_shift: Optional[float] = None,
        seed=DEFAULT_SEED,
        shear: Tuple[float, float] = (0.0, 0.0),
    ):
        """Initializes default sampling function with shear.

        Args:
            max_number: Defined in parent class.
            min_number: Defined in parent class.
            stamp_size: Defined in parent class.
            max_shift: Defined in parent class.
            seed: Defined in parent class.
            shear: Constant (g1,g2) shear to apply to every galaxy.
        """
        super().__init__(max_number, min_number, stamp_size, max_shift, seed)
        self.shear = shear

    def __call__(self, table: Table, **kwargs) -> Table:
        """Same as corresponding function for `DefaultSampling` but adds shear to output tables."""
        blend_table = super().__call__(table)
        blend_table["g1"] = self.shear[0]
        blend_table["g2"] = self.shear[1]
        return blend_table


class PairSampling(SamplingFunction):
    """Sampling function for pairs of galaxies. Picks one centered bright galaxy and second dim.

    The bright galaxy is centered at the center of the stamp and the dim galaxy is shifted.
    The bright galaxy is chosen with magnitude less than `bright_cut` and the dim galaxy
    is chosen with magnitude cut larger than `bright_cut` and less than `dim_cut`. The cuts
    can be customized by the user at initialization.

    """

    def __init__(
        self,
        stamp_size: float = 24.0,
        max_shift: float = Optional[None],
        mag_name: str = "i_ab",
        seed: int = DEFAULT_SEED,
        bright_cut: float = 25.3,
        dim_cut: float = 28.0,
    ):
        """Initializes the PairSampling function.

        Args:
            stamp_size: Size of the desired stamp (in arcseconds).
            max_shift: Maximum value of shift from center. If None then its set as one-tenth the
                stamp size (in arcseconds).
            mag_name: Name of the magnitude column in the catalog to be used.
            seed: See parent class.
            bright_cut: Magnitude cut for bright galaxy. (Default: 25.3)
            dim_cut: Magnitude cut for dim galaxy. (Default: 28.0)
        """
        super().__init__(2, 1, seed)
        self.stamp_size = stamp_size
        self.max_shift = max_shift if max_shift is not None else self.stamp_size / 10.0
        self.mag_name = mag_name
        self.bright_cut = bright_cut
        self.dim_cut = dim_cut

    def __call__(self, table: Table):
        """Samples galaxies from input catalog to make blend scene."""
        if self.mag_name not in table.colnames:
            raise ValueError(f"Catalog must have '{self.mag_name}' column.")

        (q_bright,) = np.where(table[self.mag_name] <= self.bright_cut)
        (q_dim,) = np.where(
            (table[self.mag_name] > self.bright_cut) & (table[self.mag_name] <= self.dim_cut)
        )

        indexes = [np.random.choice(q_bright), np.random.choice(q_dim)]
        blend_table = table[indexes]

        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0

        x_peak, y_peak = _get_random_center_shift(1, self.max_shift, self.rng)

        blend_table["ra"][1] += x_peak
        blend_table["dec"][1] += y_peak

        _raise_error_if_out_of_bounds(blend_table["ra"], blend_table["dec"], self.stamp_size)
        return blend_table


class RandomSquareSampling(SamplingFunction):
    """Randomly selects galaxies in square region of the input catalog.

    This sampling function explicitly uses the spatial information in the input catalog to
    generate scenes of galaxies. However, blends might not always be returned as a result.
    """

    def __init__(
        self,
        max_number: int = 2,
        stamp_size: float = 24.0,
        seed: int = DEFAULT_SEED,
        max_mag: float = 25.3,
        min_mag: float = -np.inf,
        mag_name: str = "i_ab",
    ):
        """Initializes the RandomSquareSampling sampling function.

        Args:
            max_number: Defined in parent class
            stamp_size: Size of the desired stamp (arcsec).
            seed: Seed to initialize randomness for reproducibility.
            min_mag: Minimum magnitude allowed in samples
            max_mag: Maximum magnitude allowed in samples.
            mag_name: Name of the magnitude column in the catalog.
        """
        super().__init__(max_number=max_number, min_number=0, seed=seed)
        self.stamp_size = stamp_size
        self.max_number = max_number
        self.max_mag = max_mag
        self.min_mag = min_mag
        self.mag_name = mag_name

    def __call__(self, table: Table):
        """Samples galaxies from input catalog to make scene.

        We assume the input catalog has `ra` and `dec` in degrees, like CATSIM does.
        """
        # filter by magnitude
        if self.mag_name not in table.colnames:
            raise ValueError(f"Catalog must have '{self.mag_name}' column.")
        cond1 = table[self.mag_name] <= self.max_mag
        cond2 = table[self.mag_name] > self.min_mag
        cond = cond1 & cond2
        blend_table = table[cond]

        ra = blend_table["ra"]
        dec = blend_table["dec"]

        # sometimes we might have data from [0, 1] U [359:360] deg
        # in this case to make coordinates close to each other
        # we shift the 0 of ra and dec coordinates.
        ra %= 360
        dec %= 360
        ra = (ra + ra.mean()) % 360
        dec = (dec + dec.mean()) % 360

        # check size of stamp is appropriate
        ra_range = ra.max() - ra.min()
        dec_range = dec.max() - dec.min()
        size = self.stamp_size / 3600
        if size > min(ra_range, dec_range):
            raise ValueError(
                f"sample size {size:.2f} exceeds range of the catalog "
                f"({ra_range:.2f}, {dec_range:.2f})"
            )

        # sample a square region
        ra_center = np.random.uniform(ra.min() + size / 2, ra.max() - size / 2)
        dec_center = np.random.uniform(dec.min() + size / 2, dec.max() - size / 2)
        ra_min, ra_max = ra_center - size / 2, ra_center + size / 2
        dec_min, dec_max = dec_center - size / 2, dec_center + size / 2

        # get indices of galaxies in the square region
        cond_ra = (ra > ra_min) & (ra < ra_max)
        cond_dec = (dec > dec_min) & (dec < dec_max)
        indices = np.where(cond_ra & cond_dec)

        # check that number of galaxies is in [0, max_number]
        if len(indices) > self.max_number:
            raise ValueError(
                "`max_number` of galaxies exceeded, decrease the stamp size, or "
                "increase `max_number`."
            )

        # recenter catalog 'ra' and 'dec' to the center of the stamp
        blend_table = blend_table[indices]
        blend_table["ra"] = ra[indices] - ra_center
        blend_table["dec"] = dec[indices] - dec_center

        # finally, convert to arcsec
        blend_table["ra"] *= 3600
        blend_table["dec"] *= 3600

        return blend_table
