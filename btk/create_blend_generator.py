class BlendGenerator:
    def __init__(
        self,
        catalog,
        sampling_function,
        batch_size=8,
        verbose=False,
        shifts=None,
        indexes=None,
    ):
        """Generates a list of blend catalogs of length batch_size. Each blend
        catalog has entries numbered between 1 and max_number, corresponding
        to overlapping objects in the blend.

        Args:
            catalog (astropy.Table.table): CatSim-like catalog.
            sampling_function (btk.sampling_functions.SamplingFunction): An object that
                                                                         return samples
                                                                         from the catalog.
            batch_size (int): Size of batches returned.
            verbose: Whether to print additional information.
            shifts (list): Contains arbitrary shifts to be applied instead of
                           random shifts. Must be of length batch_size. Must be used
                           with ids.
            indexes (list): Contains the ids of the galaxies to use in the stamp.
                        Must be of length batch_size. Must be used with shifts.
        """
        self.catalog = catalog
        self.batch_size = batch_size
        self.verbose = verbose
        self.sampling_function = sampling_function
        self.shifts = shifts
        self.indexes = indexes

        if not hasattr(sampling_function, "max_number"):
            raise AttributeError(
                "Please change your custom sampling function to have "
                "an attribute 'max_number'."
            )

        self.max_number = self.sampling_function.max_number

    def __iter__(self):
        return self

    def __next__(self):
        try:
            blend_catalogs = []
            for i in range(self.batch_size):
                if self.shifts is not None and self.indexes is not None:
                    blend_catalog = self.sampling_function(
                        self.catalog, shifts=self.shifts[i], indexes=self.indexes[i]
                    )
                else:
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
