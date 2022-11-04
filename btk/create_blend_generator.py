"""Contains class `BlendGenerator` to combine entries from a given catalog into blends."""


class BlendGenerator:
    """Class that uses a catalog and a sampling function to return blend information in batches."""

    def __init__(
        self,
        catalog,
        sampling_function,
        batch_size=8,
        shifts=None,
        indexes=None,
        verbose=False,
    ):
        """Initializes the BlendGenerator.

        Args:
            catalog (btk.catalog.Catalog): BTK Catalog object
            sampling_function (btk.sampling_functions.SamplingFunction): An object that
                                                                         return samples
                                                                         from the catalog.
            batch_size (int): Size of batches returned.
            shifts (list): Contains arbitrary shifts to be applied instead of
                           random shifts. Must be of length batch_size. Must be used
                           with indexes. Used mostly for internal testing purposes.
            indexes (list): Contains the ids of the galaxies to use in the stamp.
                        Must be of length batch_size. Must be used with shifts.
                        Used mostly for internal testing purposes.
            verbose (bool): Whether to print additional information.
        """
        self.catalog = catalog
        self.batch_size = batch_size
        self.verbose = verbose
        self.sampling_function = sampling_function
        self.shifts = shifts
        self.indexes = indexes

        if not hasattr(sampling_function, "max_number"):
            raise AttributeError(
                "Please change your custom sampling function to have an attribute 'max_number'."
            )

        if self.catalog.name not in self.sampling_function.compatible_catalogs:
            raise AttributeError(
                "Your catalog and sampling functions are not compatible with each other."
            )

        self.max_number = self.sampling_function.max_number
        self.min_number = self.sampling_function.min_number

    def __iter__(self):
        """Returns an iterable which is the object itself."""
        return self

    def __next__(self):
        """Generates a list of blend tables of len batch_size.

        Each blend table has entries numbered between min_number and max_number, corresponding
        to overlapping objects in the blend.

        Returns:
            blend_tables (list): a list of astropy tables, each one corresponding to a blend.
        """
        try:
            blend_tables = []
            for i in range(self.batch_size):
                if self.shifts is not None and self.indexes is not None:
                    blend_table = self.sampling_function(
                        self.catalog.table,
                        shifts=self.shifts[i],
                        indexes=self.indexes[i],
                    )
                else:
                    blend_table = self.sampling_function(self.catalog.table)
                if len(blend_table) > self.max_number:
                    raise ValueError(
                        f"Number of objects per blend must be "
                        f"less than max_number: {len(blend_table)} <= {self.max_number}"
                    )
                if len(blend_table) < self.min_number:
                    raise ValueError(
                        f"Number of objects per blend must be "
                        f"greater than min_number: {len(blend_table)} >= {self.min_number}"
                    )
                blend_tables.append(blend_table)
            return blend_tables

        except (GeneratorExit, KeyboardInterrupt):
            raise
