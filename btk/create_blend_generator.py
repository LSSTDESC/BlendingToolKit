from .sampling_functions import DefaultSampling


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
