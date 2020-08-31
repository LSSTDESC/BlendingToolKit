import warnings
import numpy as np
from abc import ABC, abstractmethod


class SamplingFunction(ABC):
    def __init__(self, max_number):
        """Class representing sampling functions to sample input catalog from which to draw
        blends.

        Args:
            max_number (int): maximum number of catalog entries returned from sample.
        """
        self.max_number = max_number

    @abstractmethod
    def __call__(self, catalog):
        """Returns a sample from the catalog with at most self.max_number of objects."""
        pass


class DefaultSampling(SamplingFunction):
    def __init__(self, max_number=4, stamp_size=24.0, maxshift=None):
        """
        Default sampling function used for producing blend catalogs.
        Args:
            max_number (int): Defined in parent class
            stamp_size (float):
            maxshift (float): Magnitude of maximum value of shift. If None then it
                             is set as one-tenth the stamp size. In arcseconds.
        """
        super().__init__(max_number)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0

    def get_random_center_shift(self, num_objects):
        """Returns random shifts in x and y coordinates between + and - max-shift
        in arcseconds.

        Args:
            num_objects(int): Number of x and y shifts to return.

        """
        dx = np.random.uniform(-self.maxshift, self.maxshift, size=num_objects)
        dy = np.random.uniform(-self.maxshift, self.maxshift, size=num_objects)
        return dx, dy

    def __call__(self, catalog):
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
            catalog: CatSim-like catalog from which to sample galaxies.

        Returns:
            Catalog with entries corresponding to one blend.
        """
        number_of_objects = np.random.randint(1, self.max_number + 1)
        (q,) = np.where(catalog["i_ab"] <= 25.3)
        blend_catalog = catalog[np.random.choice(q, size=number_of_objects)]
        blend_catalog["ra"], blend_catalog["dec"] = 0.0, 0.0
        dx, dy = self.get_random_center_shift(number_of_objects)
        blend_catalog["ra"] += dx
        blend_catalog["dec"] += dy

        if np.any(blend_catalog["ra"] > self.stamp_size / 2.0) or np.any(
            blend_catalog["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_catalog
