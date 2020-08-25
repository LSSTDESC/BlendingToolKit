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
    def __init__(self, max_number, stamp_size, maxshift=None):
        """
        Default sampling function used for producing blend catalogs.
        Args:
            max_number (int): Defined in parent class
            stamp_size:
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


class BlendGenerator:
    def __init__(self, catalog, batch_size=8, sampling_function=None, verbose=False):
        """Generates a list of blend catalogs of length batch_size. Each blend
           catalog has entries numbered between 1 and max_number, corresponding
           to overlapping objects in the blend.
        """
        self.catalog = catalog
        self.batch_size = batch_size
        self.sampling_function = self.get_sampling_function(sampling_function)
        self.max_number = self.sampling_function.max_number
        self.verbose = verbose

    def get_sampling_function(self, sampling_function):
        if not sampling_function:
            if self.verbose:
                print(
                    "Blends sampled from the catalog with the default random sampling "
                    "function "
                )
            sampling_function = DefaultSampling(max_number=4, stamp_size=24)
        if not hasattr(sampling_function, "max_number"):
            raise AttributeError("Sampling function must have attribute 'max_number'.")
        return sampling_function

    def __iter__(self):
        return self

    def __next__(self):
        try:
            blend_catalogs = []
            for i in range(self.batch_size):
                blend_catalog = self.sampling_function(self.catalog)
                if len(blend_catalog) > self.max_number:
                    raise ValueError(
                        "Number of objects per blend must be "
                        "less than max_number: {0} <= {1}".format(
                            len(blend_catalog), self.max_number
                        )
                    )
                blend_catalogs.append(blend_catalog)
            return blend_catalogs

        except (GeneratorExit, KeyboardInterrupt):
            raise
