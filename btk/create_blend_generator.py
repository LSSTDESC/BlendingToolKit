class BlendGenerator:
    def __init__(
        self,
        catalog,
        sampling_function,
        batch_size=8,
        shifts=None,
        indexes=None,
        verbose=False,
    ):
        """Generates a list of blend tables of length batch_size. Each blend
        table has entries numbered between 1 and max_number, corresponding
        to overlapping objects in the blend.

        Args:
            catalog (btk.catalog.Catalog): BTK Catalog object
            sampling_function (btk.sampling_functions.SamplingFunction): An object that
                                                                         return samples
                                                                         from the catalog.
            batch_size (int): Size of batches returned.
            shifts (list): Contains arbitrary shifts to be applied instead of
                           random shifts. Must be of length batch_size. Must be used
                           with ids.
            indexes (list): Contains the ids of the galaxies to use in the stamp.
                        Must be of length batch_size. Must be used with shifts.
            verbose: Whether to print additional information.
        """
        self.catalog = catalog
        self.batch_size = batch_size
        self.verbose = verbose
        self.sampling_function = sampling_function
        self.shifts = shifts
        self.indexes = indexes

        if not hasattr(sampling_function, "max_number"):
            raise AttributeError(
<<<<<<< HEAD
<<<<<<< HEAD
                "Please change your custom sampling function to have an attribute 'max_number'."
=======
                "Please change your custom sampling function to have " "an attribute 'max_number'."
>>>>>>> added flake8 as a pre-commit hook with custom arguments
=======
                "Please change your custom sampling function to have an attribute 'max_number'."
>>>>>>> cleaning up some newly merged lines of text
            )

        if self.catalog.name not in self.sampling_function.compatible_catalogs:
            raise AttributeError(
<<<<<<< HEAD
<<<<<<< HEAD
                "Your catalog and sampling functions are not compatible with each other."
=======
                "Your catalog and sampling functions are not " "compatible with each other."
>>>>>>> added flake8 as a pre-commit hook with custom arguments
=======
                "Your catalog and sampling functions are not compatible with each other."
>>>>>>> cleaning up some newly merged lines of text
            )

        self.max_number = self.sampling_function.max_number

    def __iter__(self):
        return self

    def __next__(self):
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
                blend_tables.append(blend_table)
            return blend_tables

        except (GeneratorExit, KeyboardInterrupt):
            raise
